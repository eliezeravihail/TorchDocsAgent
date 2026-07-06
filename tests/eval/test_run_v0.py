"""eval/run_v0.py — the grounded-api-rate metric (share of symbols in the index)."""

from types import SimpleNamespace

from eval.run_v0 import _grounded_api_rate


class FakeConn:
    """Reports a tsvector hit only for symbols in `present`."""

    def __init__(self, present):
        self.present = set(present)

    def execute(self, sql, params):
        symbol = params[0]
        return SimpleNamespace(fetchone=lambda: (1,) if symbol in self.present else None)


def test_rate_is_none_for_no_symbols():
    assert _grounded_api_rate(FakeConn([]), []) is None


def test_rate_is_fraction_of_symbols_found():
    conn = FakeConn({"torch.optim.SGD"})
    rate = _grounded_api_rate(conn, ["torch.optim.SGD", "torch.made.Up"])
    assert rate == 0.5


def test_rate_is_one_when_all_present():
    conn = FakeConn({"torch.nn.Linear", "torch.relu"})
    assert _grounded_api_rate(conn, ["torch.nn.Linear", "torch.relu"]) == 1.0
