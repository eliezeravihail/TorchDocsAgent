"""Input guardrail — one check on the user's raw question at the trust boundary.

Runs ONCE on the incoming user question (in app.main.respond / scripts.ask),
never on the internal planner / tool / translation / repair calls that form an
answer — those operate on an already-vetted question and on trusted docs
content, so re-checking them would waste CPU and could false-block on doc text.

Two cheap checks, short-circuited cheapest-first:
  1. length      — reject oversized pastes before anything else
  2. injection   — a local CPU classifier (Meta Llama Prompt Guard 2) flags
                   prompt-injection / jailbreak attempts ("ignore your rules …")
  3. topicality  — the question must actually retrieve something close in the
                   PyTorch docs, else it's someone using the bot as a free
                   general-purpose LLM

Fail-open by design: if the classifier can't load (e.g. a gated model whose
license wasn't accepted) or retrieval errors, we log and ALLOW — the guard can
never take the app down. Toggle everything off with TORCHDOCS_GUARD=0.

Deps (classifier / distance functions) are injectable so tests run without the
model download or a live database.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

# Default is Meta's tiny multilingual injection classifier (covers Hebrew too).
# It is a gated model — accept its license on Hugging Face and give the Space an
# HF token, or point this at a non-gated model, e.g.
# protectai/deberta-v3-base-prompt-injection-v2. If it can't load, the injection
# check simply no-ops (fail-open) and topicality still runs.
DEFAULT_PROMPTGUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-22M"

# cosine distance (pgvector <=>, 0=identical..2=opposite). A question whose
# nearest doc chunk is farther than this is treated as off-topic. CONSERVATIVE
# default — calibrate against the live index (see the module docstring / plan)
# before tightening, so real PyTorch questions are never blocked.
DEFAULT_TOPICALITY_MAX_DISTANCE = 0.80

REFUSAL_INJECTION = (
    "I'm a PyTorch-documentation assistant and can't act on instructions that "
    "try to override that. Ask me a PyTorch question and I'll help."
)
REFUSAL_OFFTOPIC = (
    "I only answer questions grounded in the PyTorch documentation — try asking "
    "about a PyTorch API, concept, or usage pattern."
)
REFUSAL_TOO_LONG = (
    "That question is too long for me to handle. Please shorten it to a focused "
    "PyTorch question."
)


class Verdict(NamedTuple):
    ok: bool
    reason: str = ""  # "" | "too_long" | "injection" | "off_topic"
    message: str = ""  # user-facing refusal when not ok


_OK = Verdict(True)


def _enabled() -> bool:
    return os.environ.get("TORCHDOCS_GUARD", "1") != "0"


# --- prompt-injection classifier (local, CPU) --------------------------------


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline

    model = os.environ.get("TORCHDOCS_PROMPTGUARD_MODEL", DEFAULT_PROMPTGUARD_MODEL)
    print(f"[guard] loading injection classifier {model} (CPU)", flush=True)
    return pipeline("text-classification", model=model, device=-1, truncation=True, max_length=512)


# labels the various models use for "not an attack"; anything else = attack
_BENIGN_LABELS = {"BENIGN", "SAFE", "NEGATIVE", "LABEL_0", "0"}


def _injection_score(question: str) -> float:
    """P(prompt injection / jailbreak) in [0, 1] from the classifier."""
    result = _classifier()(question)[0]
    label = str(result["label"]).upper()
    score = float(result["score"])
    return (1.0 - score) if label in _BENIGN_LABELS else score


def _score_safe(score_fn: Callable[[str], float], question: str) -> float | None:
    try:
        return score_fn(question)
    except Exception as exc:  # noqa: BLE001 — fail-open: an unavailable classifier must not block
        print(f"[guard] injection classifier unavailable ({exc}); allowing", flush=True)
        return None


# --- topicality (reuses the retrieval index) ---------------------------------


def _is_on_topic(question: str, distance_fn: Callable[[str], float | None] | None) -> bool:
    if distance_fn is None:
        from index.retrieve import top_distance

        distance_fn = top_distance
    try:
        distance = distance_fn(question)
    except Exception as exc:  # noqa: BLE001 — fail-open on any retrieval error
        print(f"[guard] topicality check skipped ({exc}); allowing", flush=True)
        return True
    if distance is None:  # empty index — a deploy problem, not the user's fault
        return True
    max_distance = float(
        os.environ.get("TORCHDOCS_TOPICALITY_MAX_DISTANCE", DEFAULT_TOPICALITY_MAX_DISTANCE)
    )
    return distance <= max_distance


def guard(
    question: str,
    *,
    injection_score_fn: Callable[[str], float] | None = None,
    distance_fn: Callable[[str], float | None] | None = None,
) -> Verdict:
    """Vet one user question. Returns Verdict(ok=True) to proceed, else a refusal."""
    if not _enabled():
        return _OK

    max_chars = int(os.environ.get("TORCHDOCS_MAX_QUESTION_CHARS", "2000"))
    if len(question) > max_chars:
        print(f"[guard] blocked over-long question ({len(question)} chars)", flush=True)
        return Verdict(False, "too_long", REFUSAL_TOO_LONG)

    threshold = float(os.environ.get("TORCHDOCS_PROMPTGUARD_THRESHOLD", "0.5"))
    score = _score_safe(injection_score_fn or _injection_score, question)
    if score is not None and score >= threshold:
        print(f"[guard] blocked injection (score={score:.2f})", flush=True)
        return Verdict(False, "injection", REFUSAL_INJECTION)

    if not _is_on_topic(question, distance_fn):
        print("[guard] blocked off-topic question", flush=True)
        return Verdict(False, "off_topic", REFUSAL_OFFTOPIC)

    return _OK


def warm_up() -> None:
    """Preload the classifier so the first guarded question isn't slow."""
    if _enabled():
        _classifier()
