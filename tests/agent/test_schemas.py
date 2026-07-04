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

from agent.schemas import Answer


def test_answer_round_trip():
    data = {
        "explanation": "torch.zeros(3) creates a 1-D tensor of length 3 filled with zeros.",
        "symbols_referenced": ["torch.zeros"],
        "torch_version": "2.7.0",
    }

    model = Answer.model_validate(data)

    assert model.model_dump() == data
