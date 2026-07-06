from types import SimpleNamespace

import pytest

from agent.llm import GenerationError, answer_question
from agent.schemas import Answer

GOOD_PAYLOAD = {
    "answer_md": "Use SGD.",
    "symbols_used": [],
    "torch_version": "2.12",
}


# --- Gemini path (default provider) -----------------------------------------


class FakeGeminiClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def _gemini_response(parsed=None, text=""):
    return SimpleNamespace(parsed=parsed, text=text)


def test_gemini_valid_answer_first_try():
    client = FakeGeminiClient([_gemini_response(parsed=Answer(**GOOD_PAYLOAD))])
    answer = answer_question("q", provider="gemini", client=client)
    assert answer.answer_md == "Use SGD."
    assert len(client.requests) == 1


def test_gemini_parses_raw_json_when_parsed_missing():
    client = FakeGeminiClient([_gemini_response(text=Answer(**GOOD_PAYLOAD).model_dump_json())])
    answer = answer_question("q", provider="gemini", client=client)
    assert answer.torch_version == "2.12"


def test_gemini_broken_reply_triggers_one_repair():
    client = FakeGeminiClient(
        [
            _gemini_response(text="not json at all"),
            _gemini_response(parsed=Answer(**GOOD_PAYLOAD)),
        ]
    )
    answer = answer_question("q", provider="gemini", client=client)
    assert answer.answer_md == "Use SGD."
    assert len(client.requests) == 2
    assert "not valid JSON" in client.requests[1]["contents"]


def test_gemini_broken_twice_raises_clean_error():
    client = FakeGeminiClient(
        [_gemini_response(text="junk"), _gemini_response(text="more junk")]
    )
    with pytest.raises(GenerationError, match="after repair"):
        answer_question("q", provider="gemini", client=client)


def test_unknown_provider_raises(monkeypatch):
    # with no other provider keys set, an unknown provider has nothing to fall
    # back to → the "unknown provider" reason surfaces
    for key in ("OPENAI_COMPAT_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(GenerationError, match="unknown provider"):
        answer_question("q", provider="cohere")


# --- OpenAI-compatible path ---------------------------------------------------


class FakeOpenAIClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content = self._replies.pop(0)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_compat_valid_answer():
    client = FakeOpenAIClient([Answer(**GOOD_PAYLOAD).model_dump_json()])
    answer = answer_question("q", provider="openai-compat", client=client)
    assert answer.answer_md == "Use SGD."
    assert client.requests[0]["response_format"] == {"type": "json_object"}


def test_openai_compat_falls_back_to_next_model_on_429(monkeypatch):
    import openai

    monkeypatch.setenv("TORCHDOCS_OPENAI_COMPAT_MODEL", "model-a,model-b")

    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        if kwargs["model"] == "model-a":
            resp = SimpleNamespace(status_code=429, headers={}, request=SimpleNamespace())
            raise openai.RateLimitError("429", response=resp, body=None)
        message = SimpleNamespace(content=Answer(**GOOD_PAYLOAD).model_dump_json())
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = answer_question("q", provider="openai-compat", client=client, retries=1)
    assert answer.answer_md == "Use SGD."  # model-b answered after model-a was rate-limited


def test_openai_compat_repair_then_error():
    client = FakeOpenAIClient(["junk", "still junk"])
    with pytest.raises(GenerationError, match="after repair"):
        answer_question("q", provider="openai-compat", client=client)
    assert len(client.requests) == 2


def test_raw_completion_uses_first_split_model_not_raw_chain(monkeypatch):
    # regression: _raw_completion (planner/translation) once sent the whole
    # comma-joined env string as a single model name, breaking every host
    from agent.llm import _raw_completion

    monkeypatch.setenv("TORCHDOCS_OPENAI_COMPAT_MODEL", "model-a,model-b")
    seen = []

    def create(**kwargs):
        seen.append(kwargs["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    out = _raw_completion("q", system="s", provider="openai-compat", client=client)
    assert out == "hi"
    assert seen == ["model-a"]  # first model of the split chain
    assert "," not in seen[0]


def test_answer_question_falls_back_to_next_provider(monkeypatch):
    # the self-healing path: the primary provider is unreachable, so the answer
    # falls through to another provider whose key is set (no user action needed)
    import agent.llm as llm

    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-x")
    monkeypatch.setenv("GEMINI_API_KEY", "g-x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TORCHDOCS_PROVIDER", raising=False)

    tried = []

    def fake_dispatch(provider, question, system, client, retries, timeout):
        tried.append(provider)
        if provider == "openai-compat":
            raise GenerationError("Connection error.")
        return Answer(**GOOD_PAYLOAD)

    monkeypatch.setattr(llm, "_dispatch", fake_dispatch)
    answer = answer_question("q")  # no explicit client → fallback chain runs
    assert answer.answer_md == "Use SGD."
    assert tried == ["openai-compat", "gemini"]  # primary failed, healed to gemini


def test_compat_client_rejects_schemeless_base_url(monkeypatch):
    # a base_url secret missing https:// otherwise surfaces as an opaque
    # "Connection error"; fail fast with the offending value instead
    from agent.llm import _compat_client

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-test")
    with pytest.raises(GenerationError, match="full http"):
        _compat_client(None, timeout=1.0)


# --- Anthropic path ----------------------------------------------------------


def _anthropic_response(payload):
    block = SimpleNamespace(type="tool_use", name="submit_answer", input=payload, id="tu_1")
    return SimpleNamespace(content=[block])


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def test_anthropic_valid_answer_first_try():
    client = FakeAnthropicClient([_anthropic_response(GOOD_PAYLOAD)])
    answer = answer_question("q", provider="anthropic", client=client)
    assert answer.answer_md == "Use SGD."
    assert len(client.requests) == 1


def test_anthropic_broken_payload_triggers_one_repair():
    broken = {"symbols_used": "not-a-list"}
    client = FakeAnthropicClient(
        [_anthropic_response(broken), _anthropic_response(GOOD_PAYLOAD)]
    )
    answer = answer_question("q", provider="anthropic", client=client)
    assert answer.torch_version == "2.12"
    assert len(client.requests) == 2
    repair_content = client.requests[1]["messages"][-1]["content"]
    assert repair_content[0]["type"] == "tool_result"
    assert repair_content[0]["is_error"] is True


def test_anthropic_broken_twice_raises_clean_error():
    broken = {"symbols_used": "not-a-list"}
    client = FakeAnthropicClient([_anthropic_response(broken), _anthropic_response(broken)])
    with pytest.raises(GenerationError, match="after repair"):
        answer_question("q", provider="anthropic", client=client)


def test_anthropic_no_tool_call_raises():
    client = FakeAnthropicClient([SimpleNamespace(content=[SimpleNamespace(type="text")])])
    with pytest.raises(GenerationError, match="no submit_answer"):
        answer_question("q", provider="anthropic", client=client)
