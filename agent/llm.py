"""LLM wrapper: one question in, one validated Answer out.

M1 baseline — no retrieval yet. The model answers from its own knowledge,
which is exactly what eval/hallucinations.md measures before M2 adds
grounding. Direct Anthropic SDK for now; LiteLLM takes over in M3.
"""

from __future__ import annotations

import os
import time

import anthropic
from pydantic import ValidationError

from agent.schemas import Answer

MODEL = os.environ.get("TORCHDOCS_MODEL", "claude-sonnet-5")

SYSTEM = (
    "You are a PyTorch documentation assistant. Answer the user's question in "
    "clear markdown, embedding short illustrative code snippets where helpful. "
    "List every PyTorch API symbol your answer relies on in symbols_used, and "
    "set torch_version to the PyTorch version your answer targets. "
    "If you are not sure something exists, say so rather than inventing an API."
)

_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer to the user's question.",
    "input_schema": Answer.model_json_schema(),
}


class GenerationError(RuntimeError):
    """The model could not produce a schema-valid answer."""


def _call(client: anthropic.Anthropic, messages: list[dict], timeout: float):
    return client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM,
        tools=[_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=messages,
        timeout=timeout,
    )


def _extract_tool_use(response):
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_answer":
            return block
    raise GenerationError("model returned no submit_answer tool call")


def answer_question(
    question: str,
    *,
    client: anthropic.Anthropic | None = None,
    retries: int = 3,
    timeout: float = 120.0,
) -> Answer:
    """Ask one question, return a validated Answer.

    Transport errors: up to `retries` attempts with exponential backoff.
    Schema errors: one repair round with the validation message, then a
    clean GenerationError.
    """
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise GenerationError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic()

    messages: list[dict] = [{"role": "user", "content": question}]

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = _call(client, messages, timeout)
            break
        except anthropic.APIError as exc:
            last_exc = exc
            time.sleep(2**attempt)
    else:
        raise GenerationError(f"LLM call failed after {retries} attempts: {last_exc}")

    block = _extract_tool_use(response)
    try:
        return Answer.model_validate(block.input)
    except ValidationError as exc:
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "is_error": True,
                        "content": (
                            f"Schema validation failed: {exc}. "
                            "Call submit_answer again with a corrected payload."
                        ),
                    }
                ],
            }
        )
        response = _call(client, messages, timeout)
        repaired = _extract_tool_use(response)
        try:
            return Answer.model_validate(repaired.input)
        except ValidationError as exc2:
            raise GenerationError(f"schema validation failed after repair: {exc2}") from exc2
