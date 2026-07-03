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

import json

import pytest

from agent.llm import GenerationError, generate_code

VALID_ANSWER = {
    "code": "import torch\nx = torch.zeros(3)",
    "explanation": "Creates a zero tensor of length 3.",
    "symbols_used": ["torch.zeros"],
    "torch_version": "2.7.0",
}


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeModels:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


class FakeClient:
    def __init__(self, replies):
        self.models = FakeModels(replies)


def test_generate_code_success_first_try():
    client = FakeClient([json.dumps(VALID_ANSWER)])

    result = generate_code("what does nn.Dropout do?", client=client)

    assert result.code == VALID_ANSWER["code"]
    assert result.symbols_used == ["torch.zeros"]
    assert client.models.calls == 1


def test_generate_code_repairs_broken_json_once():
    client = FakeClient(["not valid json {{{", json.dumps(VALID_ANSWER)])

    result = generate_code("what does nn.Dropout do?", client=client)

    assert result.torch_version == "2.7.0"
    assert client.models.calls == 2


def test_generate_code_gives_up_after_one_repair_attempt():
    client = FakeClient(["not valid json {{{", "still not valid json {{{"])

    with pytest.raises(GenerationError):
        generate_code("what does nn.Dropout do?", client=client)

    assert client.models.calls == 2


def test_generate_code_retries_on_transient_errors(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda _: None)
    client = FakeClient([RuntimeError("network blip"), json.dumps(VALID_ANSWER)])

    result = generate_code("what does nn.Dropout do?", client=client, max_retries=3)

    assert result.code == VALID_ANSWER["code"]
    assert client.models.calls == 2


def test_generate_code_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda _: None)
    client = FakeClient([RuntimeError("down"), RuntimeError("down"), RuntimeError("down")])

    with pytest.raises(GenerationError):
        generate_code("what does nn.Dropout do?", client=client, max_retries=3)

    assert client.models.calls == 3
