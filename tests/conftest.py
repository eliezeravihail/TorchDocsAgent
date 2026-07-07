"""Suite-wide isolation for module-level state.

agent/llm.py keeps a process-wide circuit breaker (model/provider cooldowns)
and agent/guard.py caches its classifier (including a failed-load cooldown).
Both are deliberate in production — state must be shared across requests — but
they would leak between tests: a model named "model-a" cooled down by one test
would be silently skipped in the next.
"""

import pytest

import agent.guard as guard_mod
from agent.llm import reset_cooldowns


@pytest.fixture(autouse=True)
def _reset_shared_llm_state(monkeypatch):
    reset_cooldowns()
    monkeypatch.setattr(guard_mod, "_CLF", None)
    monkeypatch.setattr(guard_mod, "_CLF_RETRY_AT", 0.0)
    yield
    reset_cooldowns()
