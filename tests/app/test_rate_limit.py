"""Per-client throttle and queue backpressure on the web app."""

from types import SimpleNamespace

import pytest

from agent.schemas import Answer
from app import main


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch):
    monkeypatch.setenv("TORCHDOCS_GUARD", "0")
    monkeypatch.setattr(main, "answer_routed", lambda q, **k: Answer(answer_md="ok"))
    main._RATE_BUCKETS.clear()
    yield
    main._RATE_BUCKETS.clear()


def _request(host: str):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_one_client_is_throttled_others_are_not(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 2)

    assert "ok" in main._pipeline("q1", _request("1.2.3.4"))
    assert "ok" in main._pipeline("q2", _request("1.2.3.4"))
    # third question inside the window → throttled, workers stay free
    assert main._pipeline("q3", _request("1.2.3.4")) == main.BUSY_NOTE
    # a different client is unaffected by the noisy one
    assert "ok" in main._pipeline("q4", _request("9.9.9.9"))


def test_direct_calls_without_a_request_are_not_throttled(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    for _ in range(3):  # no request object (tests, scripts) → no client to key on
        assert "ok" in main._pipeline("q")


def test_zero_disables_the_throttle(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 0)
    for _ in range(5):
        assert "ok" in main._pipeline("q", _request("1.2.3.4"))


def test_window_expiry_frees_the_client(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    clock = {"now": 1000.0}
    monkeypatch.setattr(main.time, "monotonic", lambda: clock["now"])

    assert "ok" in main._pipeline("q1", _request("1.2.3.4"))
    assert main._pipeline("q2", _request("1.2.3.4")) == main.BUSY_NOTE
    clock["now"] += main.RATE_WINDOW + 1  # window passes → slot frees up
    assert "ok" in main._pipeline("q3", _request("1.2.3.4"))
