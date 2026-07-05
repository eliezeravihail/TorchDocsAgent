from agent.schemas import Answer
from eval.checks import run_checks

GOOD = Answer(
    answer_md=(
        "Use `torch.optim.SGD`:\n"
        "```python\n"
        "import torch\n"
        "opt = torch.optim.SGD(model.parameters(), lr=0.01)\n"
        "```\n"
    ),
    symbols_used=["torch.optim.SGD"],
    torch_version="2.12",
)


def test_good_answer_passes_all():
    assert all(v is None for v in run_checks(GOOD).values())


def test_syntax_error_caught():
    bad = GOOD.model_copy(update={"answer_md": "```python\ndef broken(:\n```"})
    assert "block 0" in run_checks(bad)["parses"]


def test_disallowed_import_caught():
    bad = GOOD.model_copy(update={"answer_md": "```python\nimport requests\n```"})
    assert "requests" in run_checks(bad)["imports"]


def test_from_import_of_torch_allowed():
    ok = GOOD.model_copy(update={"answer_md": "```python\nfrom torch import nn\n```"})
    assert run_checks(ok)["imports"] is None


def test_missing_symbol_caught():
    bad = GOOD.model_copy(update={"symbols_used": ["torch.nn.LSTM"]})
    assert "torch.nn.LSTM" in run_checks(bad)["symbols"]


def test_aliased_symbol_accepted():
    ok = GOOD.model_copy(
        update={
            "answer_md": "```python\nfrom torch import nn\nlayer = nn.Linear(3, 4)\n```",
            "symbols_used": ["torch.nn.Linear"],
        }
    )
    assert run_checks(ok)["symbols"] is None


def test_indented_code_block_parses():
    ok = GOOD.model_copy(
        update={"answer_md": "```python\n    import torch\n    x = torch.ones(2)\n```"}
    )
    assert run_checks(ok)["parses"] is None
