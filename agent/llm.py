"""LLM wrapper: one question in, one validated Answer out.

Three providers behind one dispatch: gemini, anthropic, and any
OpenAI-compatible host (OpenRouter / DeepInfra / Nebius / ...). The provider
is chosen by TORCHDOCS_PROVIDER, or auto-detected from whichever API key is
set (see default_provider). The openai-compat path is the free-tier
deployment default; anthropic is the paid production path.

Every provider returns a schema-valid Answer or raises GenerationError — the
structured-output mechanism (forced tool call / response_schema / JSON mode)
is per-provider but the contract is uniform, including one schema-repair
round. The OpenAI-compatible transport is shared by the schema and raw
(planner/translation) paths via _compat_client / _compat_complete so the two
cannot drift.
"""

from __future__ import annotations

import os
import random
import threading
import time

from pydantic import ValidationError

from agent.schemas import Answer

GEMINI_MODEL = os.environ.get("TORCHDOCS_GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL = os.environ.get("TORCHDOCS_ANTHROPIC_MODEL", "claude-sonnet-5")
# any OpenAI-compatible host (OpenRouter, DeepInfra, Nebius, ...) — see .env.example.
# comma-separated → fallback chain: if one model is rate-limited/gone, try the next.
# Default is a chain of real OpenRouter free-tier slugs (org/model:free); an
# invalid slug is the classic "OpenRouter never answers" bug — every call 404s.
# deepseek-chat-v3-0324:free was retired to paid-only (404 as of 2026-07-08) —
# a dead slug at the head of the chain costs a wasted call per request.
# hy3 leads: with OpenRouter credit its free-tier daily cap is far wider than
# gemini's 20/day, so it carries the deployment; llama-3.3-70b is the fallback.
DEFAULT_COMPAT_MODELS = (
    "tencent/hy3:free,"
    "meta-llama/llama-3.3-70b-instruct:free"
)
OPENAI_COMPAT_MODEL = os.environ.get("TORCHDOCS_OPENAI_COMPAT_MODEL", DEFAULT_COMPAT_MODELS)


def _compat_models() -> list[str]:
    # re-read env each call so a late-set TORCHDOCS_OPENAI_COMPAT_MODEL (tests,
    # some deploys) still takes effect; fall back to the import-time default
    raw = os.environ.get("TORCHDOCS_OPENAI_COMPAT_MODEL") or OPENAI_COMPAT_MODEL
    return [m.strip() for m in raw.split(",") if m.strip()]

SYSTEM = (
    "You are a PyTorch documentation assistant. Answer the user's question in "
    "clear markdown, embedding short illustrative code snippets where helpful. "
    "List every PyTorch API symbol your answer relies on in symbols_used, and "
    "set torch_version to the PyTorch version your answer targets. "
    "If you are not sure something exists, say so rather than inventing an API."
)


class GenerationError(RuntimeError):
    """The model could not produce a schema-valid answer."""


# --- cooldowns: a process-wide circuit breaker --------------------------------

# A model/provider that just failed is skipped by OTHER calls for a short
# window instead of being re-tried (and slept on) from scratch. One question
# makes ~13 LLM calls (translation + planner steps + answer + repair), and the
# app serves many questions concurrently — without shared state, every one of
# those calls hammers the same rate-limited model first and pays the full
# retry+sleep cost before falling through. Entries expire on their own; if
# EVERYTHING is cooling down we try anyway rather than refuse to answer.
_COOLDOWN_LOCK = threading.Lock()
_COOLDOWNS: dict[str, float] = {}  # "model:x" / "provider:x" → monotonic deadline

MODEL_COOLDOWN_429 = 60.0  # free-tier rate-limit windows are about a minute
MODEL_COOLDOWN_404 = 3600.0  # retired/renamed slug — pointless to re-try soon
PROVIDER_COOLDOWN = 60.0
COOLDOWN_MAX = 900.0  # never honor a server-suggested wait beyond 15 minutes
RETRY_AFTER_SLEEP_CAP = 30.0  # sleep at most this long; longer → cool down, next model


def _set_cooldown(key: str, seconds: float) -> None:
    with _COOLDOWN_LOCK:
        _COOLDOWNS[key] = time.monotonic() + min(seconds, COOLDOWN_MAX)


def _skip_cooling(items: list[str], kind: str) -> list[str]:
    """Drop items in an active cooldown — unless that would leave nothing."""
    now = time.monotonic()
    with _COOLDOWN_LOCK:
        hot = [i for i in items if _COOLDOWNS.get(f"{kind}:{i}", 0.0) <= now]
    if hot and hot != items:
        skipped = ", ".join(i for i in items if i not in hot)
        print(f"[llm] skipping cooling-down {kind}(s): {skipped}", flush=True)
    return hot or items  # everything cooling → try anyway rather than give up


def reset_cooldowns() -> None:
    """Clear the circuit-breaker state (used by tests)."""
    with _COOLDOWN_LOCK:
        _COOLDOWNS.clear()


def _gemini_key() -> str | None:
    """Gemini key under any of the common secret names.

    Deploys name this secret inconsistently — GEMINI_API_KEY (what the SDK
    docs use), a bare GEMINI, or GOOGLE_API_KEY. Accept all so a
    correctly-set-but-mis-named secret still enables the gemini fallback
    instead of silently disabling it.
    """
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI")
        or os.environ.get("GOOGLE_API_KEY")
    )


def _configured_providers() -> list[str]:
    """Providers whose API key is present, in preference order."""
    avail = []
    if os.environ.get("OPENAI_COMPAT_API_KEY"):
        avail.append("openai-compat")
    if os.environ.get("ANTHROPIC_API_KEY"):
        avail.append("anthropic")
    if _gemini_key():
        avail.append("gemini")
    return avail


def default_provider() -> str:
    """Resolve the provider: explicit env, else whichever key is configured.

    Keeps a deploy working when TORCHDOCS_PROVIDER is forgotten — if an
    OpenRouter/OpenAI-compat key is set, use it rather than falling back to
    gemini (whose SDK may not even be installed).
    """
    explicit = os.environ.get("TORCHDOCS_PROVIDER")
    if explicit:
        return explicit
    configured = _configured_providers()
    return configured[0] if configured else "gemini"


def _provider_chain(preferred: str) -> list[str]:
    """Preferred provider first, then every other provider whose key is set.

    This is the self-healing order: if the primary host is unreachable or its
    secret is misconfigured, the answer path falls through to any other
    provider that has credentials — so one broken secret can't take the whole
    deploy down. Only used when no explicit client is passed (i.e. real calls,
    not the fake-client tests).
    """
    chain = [preferred]
    for p in _configured_providers():
        if p not in chain:
            chain.append(p)
    return chain


def _dispatch(
    provider: str, question: str, system: str, client, retries: int, timeout: float
) -> Answer:
    if provider == "gemini":
        return _answer_gemini(question, system, client, retries, timeout)
    if provider == "anthropic":
        return _answer_anthropic(question, system, client, retries, timeout)
    if provider == "openai-compat":
        return _answer_openai_compat(question, system, client, retries, timeout)
    raise GenerationError(f"unknown provider: {provider}")


def answer_question(
    question: str,
    *,
    system: str = SYSTEM,
    provider: str | None = None,
    client=None,
    retries: int = 3,
    timeout: float = 120.0,
) -> Answer:
    """Ask one question, return a validated Answer.

    Transport errors: up to `retries` attempts with backoff (longer on 429,
    the free-tier rate limit). Schema errors: one repair round with the
    validation message, then a clean GenerationError. If the primary provider
    is unreachable, fall back to any other provider that has a key set.
    """
    provider = provider or default_provider()
    if client is not None:  # explicit client → single provider, no fallback
        return _dispatch(provider, question, system, client, retries, timeout)

    chain = _skip_cooling(_provider_chain(provider), "provider")
    last_exc: Exception | None = None
    for i, p in enumerate(chain):
        try:
            answer = _dispatch(p, question, system, None, retries, timeout)
            if i > 0:  # make silent quality drift visible: who actually answered?
                print(f"[llm] answered by fallback provider {p}", flush=True)
            return answer
        except Exception as exc:  # noqa: BLE001 — resilience across providers
            last_exc = exc
            _set_cooldown(f"provider:{p}", PROVIDER_COOLDOWN)
            nxt = f"; falling back to {chain[i + 1]}" if i + 1 < len(chain) else ""
            print(f"[llm] provider {p} failed ({type(exc).__name__}: {exc}){nxt}", flush=True)
    raise GenerationError(f"all providers failed ({', '.join(chain)}): {last_exc}")


def _raw_completion(
    prompt: str,
    *,
    system: str,
    provider: str | None = None,
    client=None,
    timeout: float = 60.0,
) -> str:
    """Plain-text completion (no schema) — for short helper calls like translation.

    Every provider's failure is wrapped in GenerationError so callers such as
    the planner (_plan) can catch one exception type and degrade gracefully.
    Mirrors answer_question's self-healing: with no explicit client, an
    unreachable primary provider falls through to any other configured one.
    """
    provider = provider or default_provider()
    if client is None:
        chain = _skip_cooling(_provider_chain(provider), "provider")
        last_exc: Exception | None = None
        for i, p in enumerate(chain):
            try:
                return _raw_dispatch(
                    prompt, system=system, provider=p, client=None, timeout=timeout
                )
            except Exception as exc:  # noqa: BLE001 — resilience across providers
                last_exc = exc
                _set_cooldown(f"provider:{p}", PROVIDER_COOLDOWN)
                if i + 1 < len(chain):
                    print(f"[llm] raw provider {p} failed; trying {chain[i + 1]}", flush=True)
        raise GenerationError(f"raw completion failed across all providers: {last_exc}")
    return _raw_dispatch(prompt, system=system, provider=provider, client=client, timeout=timeout)


def _raw_dispatch(
    prompt: str,
    *,
    system: str,
    provider: str,
    client=None,
    timeout: float = 60.0,
) -> str:
    if provider == "gemini":
        from google import genai
        from google.genai import types

        if client is None:
            key = _gemini_key()
            if not key:
                raise GenerationError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=key, http_options={"timeout": int(timeout * 1000)})
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system),
            )
        except Exception as exc:  # noqa: BLE001 — normalize SDK errors for callers
            raise GenerationError(f"gemini raw completion failed: {exc}") from exc
        return getattr(response, "text", "") or ""

    if provider == "openai-compat":
        client = _compat_client(client, timeout)
        response, _ = _compat_complete(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            json_mode=False,
        )
        return response.choices[0].message.content or ""

    if provider == "anthropic":
        import anthropic

        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise GenerationError("ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic()
        try:
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 — normalize SDK errors for callers
            raise GenerationError(f"anthropic raw completion failed: {exc}") from exc
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    raise GenerationError(f"unknown provider: {provider}")


# --- OpenAI-compatible transport (shared by the raw and schema paths) --------


# One OpenAI-compatible client per (base_url, key, timeout), reused across
# requests. openai.OpenAI holds an httpx connection pool, so rebuilding it per
# call would throw away keep-alive and pay a fresh TLS handshake every LLM hop —
# wasteful once many questions are answered at once. The client is thread-safe;
# the lock only guards the one-time build.
_COMPAT_CLIENTS: dict[tuple, object] = {}
_COMPAT_CLIENTS_LOCK = threading.Lock()


def _compat_client(client, timeout: float):
    """Build (or pass through) an OpenAI-compatible client, validating base_url."""
    if client is not None:
        return client
    import openai

    base_url = os.environ.get("OPENAI_COMPAT_BASE_URL") or "https://openrouter.ai/api/v1"
    if not base_url.startswith("http"):
        # a misconfigured secret (missing scheme) otherwise surfaces as an
        # opaque "Connection error" — fail loudly with the offending value
        raise GenerationError(
            f"OPENAI_COMPAT_BASE_URL must be a full http(s) URL, got {base_url!r}"
        )
    key = os.environ.get("OPENAI_COMPAT_API_KEY")
    if not key:
        raise GenerationError("OPENAI_COMPAT_API_KEY not set")
    cache_key = (base_url, key, timeout)
    with _COMPAT_CLIENTS_LOCK:
        cached = _COMPAT_CLIENTS.get(cache_key)
        if cached is None:
            print(f"[llm] openai-compat base_url={base_url} models={_compat_models()}", flush=True)
            cached = openai.OpenAI(base_url=base_url, api_key=key, timeout=timeout)
            _COMPAT_CLIENTS[cache_key] = cached
    return cached


def _compat_complete(client, messages, *, retries: int = 3, json_mode: bool = False):
    """Run the model fallback chain with retries; return (response, json_mode).

    Walks _compat_models() so a rate-limited or retired model falls through to
    the next — the raw and schema paths both go through here, so the fallback
    behaviour is identical (the earlier raw path only ever tried one model).
    json_mode may flip off if a host rejects response_format; the updated value
    is returned so a follow-up call keeps it off.
    """
    import openai

    models = _skip_cooling(_compat_models(), "model")
    last_exc: Exception | None = None
    for model in models:  # fallback chain across configured models
        for attempt in range(retries):
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                if not getattr(response, "choices", None):
                    # OpenRouter can return HTTP 200 whose body carries an
                    # error and no choices (moderation, upstream hiccup); the
                    # SDK parses it happily and the caller then crashes on
                    # choices[0]. Count it as a failed attempt, not an answer.
                    err = getattr(response, "error", None)
                    last_exc = GenerationError(f"{model} returned no choices: {err!r}")
                    print(f"[llm] {model} returned no choices ({err!r})", flush=True)
                    continue
                if model != models[0]:  # visibility: a fallback model answered
                    print(f"[llm] answered by fallback model {model}", flush=True)
                return response, json_mode
            except openai.APIError as exc:
                last_exc = exc
                # a bare "Connection error." hides the real reason (DNS, TLS,
                # blocked egress); surface the wrapped cause so logs are useful
                cause = getattr(exc, "__cause__", None)
                detail = f" (cause: {cause!r})" if cause else ""
                print(f"[llm] {model} error: {type(exc).__name__}: {exc}{detail}", flush=True)
                status = getattr(exc, "status_code", None)
                if status == 404:  # slug retired / now paid → skip to the next model
                    _set_cooldown(f"model:{model}", MODEL_COOLDOWN_404)
                    print(f"[llm] {model} unavailable (404); trying next model")
                    break
                if json_mode and status in (400, 422):
                    json_mode = False  # host rejected JSON mode — prompt+repair covers it
                    continue
                if status == 429:
                    retry_after = _retry_after_seconds(exc)
                    # Give up on this model — and record a cooldown so other
                    # in-flight calls skip it — when retries are exhausted or
                    # the server asks for a longer wait than we're willing to
                    # hold a worker thread for (e.g. a daily-quota Retry-After
                    # of an hour must not put a request to sleep for an hour).
                    if attempt == retries - 1 or (
                        retry_after is not None and retry_after > RETRY_AFTER_SLEEP_CAP
                    ):
                        _set_cooldown(
                            f"model:{model}",
                            retry_after if retry_after is not None else MODEL_COOLDOWN_429,
                        )
                        print(f"[llm] {model} rate-limited; trying next model")
                        break  # exhausted this model → next in the chain
                    # Honor a short Retry-After; otherwise widen the window each
                    # attempt. Jitter keeps a burst of requests that all hit 429
                    # in the same second from retrying in lockstep.
                    base = retry_after if retry_after is not None else 20 * (attempt + 1)
                    time.sleep(base + random.uniform(0, 1.5))
                else:
                    time.sleep(2**attempt)
    raise GenerationError(f"LLM call failed across {len(models)} model(s): {last_exc}")


def _retry_after_seconds(exc) -> float | None:
    """Seconds the server asked us to wait, from a Retry-After header if present.

    Returns None when there is no header or it is an HTTP-date rather than a
    delta-seconds count, so the caller falls back to its own backoff.
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# --- Gemini (default: free tier) -------------------------------------------


def _answer_gemini(question: str, system: str, client, retries: int, timeout: float) -> Answer:
    from google import genai
    from google.genai import errors, types

    if client is None:
        key = _gemini_key()
        if not key:
            raise GenerationError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=key, http_options={"timeout": int(timeout * 1000)})

    config = types.GenerateContentConfig(
        system_instruction=system,
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


def _answer_openai_compat(
    question: str, system: str, client, retries: int, timeout: float
) -> Answer:
    client = _compat_client(client, timeout)

    schema_prompt = (
        f"{system}\nReply with ONLY a JSON object matching this schema:\n"
        f"{Answer.model_json_schema()}"
    )
    messages = [
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": question},
    ]

    response, json_mode = _compat_complete(client, messages, retries=retries, json_mode=True)
    reply = response.choices[0].message.content or ""
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
        response, _ = _compat_complete(client, messages, retries=retries, json_mode=json_mode)
        repaired = response.choices[0].message.content or ""
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


def _answer_anthropic(question: str, system: str, client, retries: int, timeout: float) -> Answer:
    import anthropic

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise GenerationError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic()

    def call(messages):
        return client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=system,
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
