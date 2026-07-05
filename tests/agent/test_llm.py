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


def test_unknown_provider_raises():
    with pytest.raises(GenerationError, match="unknown provider"):
        answer_question("q", provider="cohere")


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
