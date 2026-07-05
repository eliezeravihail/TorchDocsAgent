"""LLM wrapper: one question in, one validated Answer out.

M1 baseline — no retrieval yet. The model answers from its own knowledge,
which is exactly what eval/hallucinations.md measures before M2 adds
grounding.

Provider is selected by TORCHDOCS_PROVIDER (default "gemini" — free tier
covers development and eval). "anthropic" is the production path once
credits exist; LiteLLM replaces this dispatch in M3.
"""

from __future__ import annotations

import os
import time

from pydantic import ValidationError

from agent.schemas import Answer

GEMINI_MODEL = os.environ.get("TORCHDOCS_GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL = os.environ.get("TORCHDOCS_ANTHROPIC_MODEL", "claude-sonnet-5")
# any OpenAI-compatible host (DeepInfra, Nebius, OpenRouter, ...) — see .env.example
OPENAI_COMPAT_MODEL = os.environ.get(
    "TORCHDOCS_OPENAI_COMPAT_MODEL", "deepseek-ai/DeepSeek-V4-Flash"
)

SYSTEM = (
    "You are a PyTorch documentation assistant. Answer the user's question in "
    "clear markdown, embedding short illustrative code snippets where helpful. "
    "List every PyTorch API symbol your answer relies on in symbols_used, and "
    "set torch_version to the PyTorch version your answer targets. "
    "If you are not sure something exists, say so rather than inventing an API."
)


class GenerationError(RuntimeError):
    """The model could not produce a schema-valid answer."""


def answer_question(
    question: str,
    *,
    provider: str | None = None,
    client=None,
    retries: int = 3,
    timeout: float = 120.0,
) -> Answer:
    """Ask one question, return a validated Answer.

    Transport errors: up to `retries` attempts with backoff (longer on 429,
    the free-tier rate limit). Schema errors: one repair round with the
    validation message, then a clean GenerationError.
    """
    provider = provider or os.environ.get("TORCHDOCS_PROVIDER", "gemini")
    if provider == "gemini":
        return _answer_gemini(question, client, retries, timeout)
    if provider == "anthropic":
        return _answer_anthropic(question, client, retries, timeout)
    if provider == "openai-compat":
        return _answer_openai_compat(question, client, retries, timeout)
    raise GenerationError(f"unknown provider: {provider}")


# --- Gemini (default: free tier) -------------------------------------------


def _answer_gemini(question: str, client, retries: int, timeout: float) -> Answer:
    from google import genai
    from google.genai import errors, types

    if client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise GenerationError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=key, http_options={"timeout": int(timeout * 1000)})

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=Answer,
    )

    def generate(contents):
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents, config=config
                )
            except errors.APIError as exc:
                last_exc = exc
                # free-tier RPM: wait out the rate-limit window, not just backoff
                time.sleep(15 * (attempt + 1) if exc.code == 429 else 2**attempt)
        raise GenerationError(f"LLM call failed after {retries} attempts: {last_exc}")

    response = generate(question)
    answer = _parse_gemini(response)
    if answer is not None:
        return answer

    repair = (
        f"{question}\n\nYour previous reply was not valid JSON for the required "
        f"schema. Previous reply:\n{getattr(response, 'text', '')!r}\n"
        "Reply again with only a valid JSON object."
    )
    answer = _parse_gemini(generate(repair))
    if answer is None:
        raise GenerationError("schema validation failed after repair")
    return answer


def _parse_gemini(response) -> Answer | None:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, Answer):
        return parsed
    try:
        return Answer.model_validate_json(getattr(response, "text", "") or "")
    except ValidationError:
        return None


# --- OpenAI-compatible hosts (DeepInfra / Nebius / OpenRouter / ...) ---------


def _answer_openai_compat(question: str, client, retries: int, timeout: float) -> Answer:
    import openai

    if client is None:
        base_url = os.environ.get("OPENAI_COMPAT_BASE_URL")
        key = os.environ.get("OPENAI_COMPAT_API_KEY")
        if not base_url or not key:
            raise GenerationError("OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY not set")
        client = openai.OpenAI(base_url=base_url, api_key=key, timeout=timeout)

    schema_prompt = (
        f"{SYSTEM}\nReply with ONLY a JSON object matching this schema:\n"
        f"{Answer.model_json_schema()}"
    )

    json_mode = True  # dropped for hosts/models that reject response_format

    def generate(messages):
        nonlocal json_mode
        last_exc: Exception | None = None
        for attempt in range(retries):
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            try:
                return client.chat.completions.create(
                    model=OPENAI_COMPAT_MODEL, messages=messages, **kwargs
                )
            except openai.APIError as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status == 404:  # model slug gone (e.g. :free variant retired) — fail fast
                    raise GenerationError(f"model not available: {exc}") from exc
                if json_mode and status in (400, 422):
                    json_mode = False  # host rejected JSON mode — prompt+repair covers it
                    continue
                time.sleep(20 * (attempt + 1) if status == 429 else 2**attempt)
        raise GenerationError(f"LLM call failed after {retries} attempts: {last_exc}")

    messages = [
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": question},
    ]
    reply = generate(messages).choices[0].message.content or ""
    try:
        return Answer.model_validate_json(reply)
    except ValidationError as exc:
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": f"That was not valid JSON for the schema: {exc}. "
                "Reply again with only a corrected JSON object.",
            }
        )
        repaired = generate(messages).choices[0].message.content or ""
        try:
            return Answer.model_validate_json(repaired)
        except ValidationError as exc2:
            raise GenerationError(f"schema validation failed after repair: {exc2}") from exc2


# --- Anthropic (production path, needs API credits) -------------------------

_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer to the user's question.",
    "input_schema": Answer.model_json_schema(),
}


def _answer_anthropic(question: str, client, retries: int, timeout: float) -> Answer:
    import anthropic

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise GenerationError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic()

    def call(messages):
        return client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=[_ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "submit_answer"},
            messages=messages,
            timeout=timeout,
        )

    messages: list[dict] = [{"role": "user", "content": question}]

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = call(messages)
            break
        except anthropic.APIError as exc:
            last_exc = exc
            # free-tier RPM window is a minute — wait it out on 429
            is_rate_limit = getattr(exc, "status_code", None) == 429
            time.sleep(20 * (attempt + 1) if is_rate_limit else 2**attempt)
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
        repaired = _extract_tool_use(call(messages))
        try:
            return Answer.model_validate(repaired.input)
        except ValidationError as exc2:
            raise GenerationError(f"schema validation failed after repair: {exc2}") from exc2


def _extract_tool_use(response):
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_answer":
            return block
    raise GenerationError("model returned no submit_answer tool call")
