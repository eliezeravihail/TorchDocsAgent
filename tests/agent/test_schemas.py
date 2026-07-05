from agent.schemas import Answer


def test_round_trip():
    original = {
        "answer_md": "Use `torch.optim.SGD`.\n```python\nimport torch\n```",
        "symbols_used": ["torch.optim.SGD"],
        "torch_version": "2.12",
    }
    assert Answer.model_validate(original).model_dump() == original


def test_defaults():
    answer = Answer(answer_md="hi")
    assert answer.symbols_used == []
    assert answer.torch_version == "unknown"
