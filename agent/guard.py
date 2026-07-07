"""Input guardrail — one check on the user's raw question at the trust boundary.

Runs ONCE on the incoming user question (in app.main.respond / scripts.ask),
never on the internal planner / tool / repair calls that form an answer —
those operate on an already-vetted question and on trusted docs content.

Two cheap checks, short-circuited cheapest-first:
  1. length      — reject oversized pastes before anything else
  2. topicality  — translate the question to English (the corpus and embedder
                   are English-only), embed it, and require its nearest doc
                   chunk to be within a calibrated cosine distance.

Membership in the docs' embedding space IS the policy: this app answers
PyTorch questions, full stop. An off-topic request and a prompt-injection
("ignore your rules and …") both land far from the corpus and get the same
refusal — no dedicated injection classifier needed (an earlier design used
one; it cost an extra model and still missed injections wrapped in on-topic
questions). What passes the gate is safe regardless, because of the grounding
contract downstream: answers come only from retrieved doc sections, citations
are validated against the provided context, code is statically checked, and
the agent's tools have no side effects.

The translator is the one LLM this check trusts, so it is prompt-hardened
(agent/translate.py): delimited input framed as data, embedded instructions
translated literally, and sanity bounds on the output. Its result is cached,
so the seed search reuses the SAME translation — the guard adds no extra LLM
call.

Fail-open by design: if translation or retrieval errors, we log and ALLOW —
the guard can never take the app down. Toggle off with TORCHDOCS_GUARD=0.

Deps (translate / distance functions) are injectable so tests run without an
LLM or a live database.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import NamedTuple

# cosine distance (pgvector <=>, 0=identical..2=opposite). A question whose
# nearest doc chunk is farther than this is treated as off-topic.
#
# CALIBRATED against the live index (scripts/calibrate_guard.py, 2026-07-07):
#   on-topic (v0 eval, 15q):     0.143–0.305
#   borderline ML-adjacent (en): 0.214–0.255  (allowed — the docs can serve them)
#   off-topic incl. injections:  0.371–0.545  (all blocked at this threshold)
# 0.35 sits between the worst on-topic (0.305) and the best off-topic (0.371).
# Re-run the "Calibrate guard" workflow after major corpus changes.
DEFAULT_TOPICALITY_MAX_DISTANCE = 0.35

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
    reason: str = ""  # "" | "too_long" | "off_topic"
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


def _is_on_topic(
    question: str,
    distance_fn: Callable[[str], float | None] | None,
    translate_fn: Callable[[str], str] | None,
) -> bool:
    from agent.translate import looks_english

    if translate_fn is None:
        from agent.translate import translate_to_english as translate_fn
    if distance_fn is None:
        from index.retrieve import top_distance

        distance_fn = top_distance
    try:
        english = translate_fn(question)
        if not looks_english(english):
            # translation degraded to the original (LLM outage / rate limit) —
            # the English-only embedder can't measure this text, and calibration
            # showed untranslated questions land in blocking range (~0.4+).
            # Fail open rather than punish the user for an outage.
            print("[guard] topicality skipped (translation unavailable); allowing", flush=True)
            return True
        distance = distance_fn(english)
    except Exception as exc:  # noqa: BLE001 — fail-open on any translation/retrieval error
        print(f"[guard] topicality check skipped ({exc}); allowing", flush=True)
        return True
    if distance is None:  # empty index — a deploy problem, not the user's fault
        return True
    max_distance = _env_float("TORCHDOCS_TOPICALITY_MAX_DISTANCE", DEFAULT_TOPICALITY_MAX_DISTANCE)
    if distance > max_distance:
        print(f"[guard] off-topic (distance={distance:.3f} > {max_distance})", flush=True)
        return False
    return True


def guard(
    question: str,
    *,
    distance_fn: Callable[[str], float | None] | None = None,
    translate_fn: Callable[[str], str] | None = None,
) -> Verdict:
    """Vet one user question. Returns Verdict(ok=True) to proceed, else a refusal."""
    if not _enabled():
        return _OK

    max_chars = int(_env_float("TORCHDOCS_MAX_QUESTION_CHARS", 2000))
    if len(question) > max_chars:
        print(f"[guard] blocked over-long question ({len(question)} chars)", flush=True)
        return Verdict(False, "too_long", REFUSAL_TOO_LONG)

    if not _is_on_topic(question, distance_fn, translate_fn):
        return Verdict(False, "off_topic", REFUSAL_OFFTOPIC)

    return _OK
