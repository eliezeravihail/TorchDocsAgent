# Copyright 2026 TorchDocsAgent contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from google import genai
from google.genai import types
from pydantic import ValidationError

from agent.schemas import Answer

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "You are a PyTorch documentation assistant. Given a question about PyTorch, "
    "explain the relevant behavior in prose and name the exact torch API symbols "
    "involved. Never write or emit code, including code blocks or fenced snippets "
    "— explain everything in words only."
)


class GenerationError(Exception):
    pass


def _client() -> genai.Client:
    api_key = os.environ["GEMINI_API_KEY"]
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(TIMEOUT_SECONDS * 1000)),
    )


def _call_model(client: genai.Client, question: str, model: str, repair_note: str | None) -> str:
    prompt = question if repair_note is None else f"{question}\n\n{repair_note}"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Answer,
        ),
    )
    return response.text


def answer_question(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
    client: genai.Client | None = None,
) -> Answer:
    client = client or _client()

    last_error: Exception | None = None
    repair_note: str | None = None

    for attempt in range(max_retries):
        try:
            raw = _call_model(client, question, model, repair_note)
        except Exception as exc:  # network / rate-limit / provider errors
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            continue

        try:
            return Answer.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            if repair_note is not None:
                # already tried one repair pass — don't loop forever on bad parsing.
                break
            repair_note = (
                "Your previous response did not match the required JSON schema "
                f"and failed with this error:\n{exc}\n"
                "Reply again with corrected JSON matching the schema exactly."
            )

    raise GenerationError(f"failed to generate a valid Answer for {question!r}") from last_error
