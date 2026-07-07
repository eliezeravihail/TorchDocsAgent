"""Input guardrail — one check on the user's raw question at the trust boundary.

Runs ONCE on the incoming user question (in app.main.respond / scripts.ask),
never on the internal planner / tool / translation / repair calls that form an
answer — those operate on an already-vetted question and on trusted docs
content, so re-checking them would waste CPU and could false-block on doc text.

Three cheap checks, short-circuited cheapest-first:
  1. length      — reject oversized pastes before anything else
  2. injection   — a local CPU classifier flags prompt-injection / jailbreak
                   attempts ("ignore your rules …")
  3. topicality  — the question must actually retrieve something close in the
                   PyTorch docs, else it's someone using the bot as a free
                   general-purpose LLM. English-only: the embedding model is
                   English-only, so a non-English question (a supported feature
                   — it is translated before retrieval) would always look "far"
                   and be false-blocked. Skipped for non-English input; the
                   grounding contract downstream keeps such answers honest.

Fail-open by design: if the classifier can't load (e.g. a gated model whose
license wasn't accepted) or retrieval errors, we log and ALLOW — the guard can
never take the app down. A failed classifier load is cached with a cooldown so
it isn't re-attempted (a slow hub download) on every question. Toggle
everything off with TORCHDOCS_GUARD=0.

Deps (classifier / distance functions) are injectable so tests run without the
model download or a live database.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

# Default is a small NON-GATED injection classifier, so a fresh deploy with no
# HF token still gets a working injection check instead of a silent no-op.
# Meta's multilingual Llama-Prompt-Guard-2-22M is a good alternative but is
# GATED — accept its license on Hugging Face and give the Space an HF token
# before pointing TORCHDOCS_PROMPTGUARD_MODEL at it.
DEFAULT_PROMPTGUARD_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

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


def _env_float(name: str, default: float) -> float:
    """An env-configured float; a malformed value logs and falls back (fail-open)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[guard] ignoring malformed {name}={raw!r}; using {default}", flush=True)
        return default


# --- prompt-injection classifier (local, CPU) --------------------------------

# Built at most once (double-checked lock, same pattern as the embedding model
# in index/embed.py). A FAILED load is also remembered: retrying would
# re-attempt a hub download on every question, adding seconds of latency while
# the check silently no-ops anyway. After the cooldown the load is attempted
# again, so a transient hub outage doesn't disable the check for the process's
# lifetime.
_CLF_LOCK = threading.Lock()
_CLF = None
_CLF_RETRY_AT = 0.0
CLF_RETRY_SECONDS = 600.0


def _build_pipeline(model: str):
    from transformers import pipeline

    return pipeline(
        "text-classification", model=model, device=-1, truncation=True, max_length=512
    )


def _classifier():
    """The injection classifier, or None while a failed load is cooling down."""
    global _CLF, _CLF_RETRY_AT
    if _CLF is not None:
        return _CLF
    with _CLF_LOCK:
        if _CLF is not None:
            return _CLF
        now = time.monotonic()
        if now < _CLF_RETRY_AT:
            return None
        model = os.environ.get("TORCHDOCS_PROMPTGUARD_MODEL", DEFAULT_PROMPTGUARD_MODEL)
        try:
            print(f"[guard] loading injection classifier {model} (CPU)", flush=True)
            _CLF = _build_pipeline(model)
        except Exception as exc:  # noqa: BLE001 — fail-open, but loudly and once per cooldown
            _CLF_RETRY_AT = now + CLF_RETRY_SECONDS
            print(
                f"[guard] injection classifier failed to load ({exc}); "
                f"injection check DISABLED for {int(CLF_RETRY_SECONDS)}s",
                flush=True,
            )
            return None
    return _CLF


# labels the various models use for "not an attack"; anything else = attack
_BENIGN_LABELS = {"BENIGN", "SAFE", "NEGATIVE", "LABEL_0", "0"}


def _injection_score(question: str) -> float | None:
    """P(prompt injection / jailbreak) in [0, 1], or None if the classifier is down."""
    clf = _classifier()
    if clf is None:
        return None
    result = clf(question)[0]
    label = str(result["label"]).upper()
    score = float(result["score"])
    return (1.0 - score) if label in _BENIGN_LABELS else score


def _score_safe(score_fn: Callable[[str], float | None], question: str) -> float | None:
    try:
        return score_fn(question)
    except Exception as exc:  # noqa: BLE001 — fail-open: an unavailable classifier must not block
        print(f"[guard] injection classifier unavailable ({exc}); allowing", flush=True)
        return None


# --- topicality (reuses the retrieval index) ---------------------------------


def _is_on_topic(question: str, distance_fn: Callable[[str], float | None] | None) -> bool:
    from agent.translate import looks_english

    if not looks_english(question):
        # the embedder is English-only: a legitimate non-English question always
        # looks "far", so distance carries no signal here. Skip, don't false-block.
        return True
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
    max_distance = _env_float("TORCHDOCS_TOPICALITY_MAX_DISTANCE", DEFAULT_TOPICALITY_MAX_DISTANCE)
    return distance <= max_distance


def guard(
    question: str,
    *,
    injection_score_fn: Callable[[str], float | None] | None = None,
    distance_fn: Callable[[str], float | None] | None = None,
) -> Verdict:
    """Vet one user question. Returns Verdict(ok=True) to proceed, else a refusal."""
    if not _enabled():
        return _OK

    max_chars = int(_env_float("TORCHDOCS_MAX_QUESTION_CHARS", 2000))
    if len(question) > max_chars:
        print(f"[guard] blocked over-long question ({len(question)} chars)", flush=True)
        return Verdict(False, "too_long", REFUSAL_TOO_LONG)

    threshold = _env_float("TORCHDOCS_PROMPTGUARD_THRESHOLD", 0.5)
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
