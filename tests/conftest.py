"""Suite-wide isolation for module-level state.

agent/llm.py keeps a process-wide circuit breaker (model/provider cooldowns)
and agent/translate.py caches default-path translations. Both are deliberate
in production — state must be shared across requests — but they would leak
between tests: a model named "model-a" cooled down by one test would be
silently skipped in the next.

The cross-encoder reranker (on by default in production) is switched off for
the unit suite — no test may download a 90MB model. Tests that exercise the
rerank path inject a fake scorer via retrieve(rerank_fn=...) instead.
"""

import os

import pytest

from agent.llm import reset_cooldowns
from agent.translate import _translate_default

os.environ.setdefault("TORCHDOCS_RERANK", "0")


@pytest.fixture(autouse=True)
def _reset_shared_llm_state():
    reset_cooldowns()
    _translate_default.cache_clear()
    yield
    reset_cooldowns()
    _translate_default.cache_clear()
