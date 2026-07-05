from types import SimpleNamespace

import pytest

from agent.llm import GenerationError, answer_question

GOOD_PAYLOAD = {
    "answer_md": "Use SGD.",
    "symbols_used": [],
    "torch_version": "2.12",
}


def _response(payload):
    block = SimpleNamespace(type="tool_use", name="submit_answer", input=payload, id="tu_1")
    return SimpleNamespace(content=[block])


class FakeClient:
    """Returns queued responses; records every request payload."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def test_valid_answer_first_try():
    client = FakeClient([_response(GOOD_PAYLOAD)])
    answer = answer_question("how do I use SGD?", client=client)
    assert answer.answer_md == "Use SGD."
    assert len(client.requests) == 1


def test_broken_payload_triggers_one_repair():
    broken = {"symbols_used": "not-a-list"}  # missing answer_md, wrong type
    client = FakeClient([_response(broken), _response(GOOD_PAYLOAD)])
    answer = answer_question("q", client=client)
    assert answer.torch_version == "2.12"
    assert len(client.requests) == 2
    # the repair turn carries the validation error back as a tool_result
    repair_content = client.requests[1]["messages"][-1]["content"]
    assert repair_content[0]["type"] == "tool_result"
    assert repair_content[0]["is_error"] is True


def test_broken_payload_twice_raises_clean_error():
    broken = {"symbols_used": "not-a-list"}
    client = FakeClient([_response(broken), _response(broken)])
    with pytest.raises(GenerationError, match="after repair"):
        answer_question("q", client=client)


def test_no_tool_call_raises():
    client = FakeClient([SimpleNamespace(content=[SimpleNamespace(type="text")])])
    with pytest.raises(GenerationError, match="no submit_answer"):
        answer_question("q", client=client)
