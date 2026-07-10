"""Suite-wide isolation for module-level state.

agent/llm.py keeps a process-wide circuit breaker (model/provider cooldowns) —
deliberate in production (state must be shared across requests) but it would
leak between tests: a model named "model-a" cooled down by one test would be
silently skipped in the next.
"""

import pytest

from agent.llm import reset_cooldowns


@pytest.fixture(autouse=True)
def _reset_shared_llm_state():
    reset_cooldowns()
    yield
    reset_cooldowns()
