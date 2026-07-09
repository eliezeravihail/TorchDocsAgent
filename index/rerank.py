"""Cross-encoder reranking: the precision stage after candidate generation.

Why a reranker, and why now: the gloss re-embed (measured 2026-07-09) moved
several expected pages INTO their kind-pool (LayerNorm: true dense rank
899 → 2; SGD: 7 → 3) but bi-encoder distance + RRF still decide the final
order. A cross-encoder reads query and candidate TOGETHER — the interaction a
bi-encoder cannot see — and fixes exactly that ordering. It cannot rescue a
page that never became a candidate (Linear at dense rank 3,412 needs deeper
index-side enrichment, not reranking); see docs/retrieval-gaps-and-improvements.md.

What the cross-encoder scores: the index stores no chunk content (pointers
only), and hydrating dozens of candidates per query is too slow for the live
app — so each candidate is scored on its metadata line: symbol + page title +
heading path + the page's committed gloss (index/glosses.jsonl). For api pages
the gloss is exactly the plain-language "what is this page for" sentence, which
is the signal a relevance judgment needs.

Runs on CPU (~90MB model, a few hundred ms for a 24-candidate slate), behind
the TORCHDOCS_RERANK kill switch so eval can measure before/after and a broken
model can be turned off without a deploy. Fail-open: any model error returns
the fused order unchanged — reranking must never take retrieval down.
"""

from __future__ import annotations

import os
import threading
from typing import Any

RERANK_MODEL = os.environ.get("TORCHDOCS_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# candidates fed to the cross-encoder: k=8 finals from a 24-candidate slate
# keeps 3× headroom for the reorder while staying sub-second on CPU
RERANK_SLATE = int(os.environ.get("TORCHDOCS_RERANK_SLATE", "24"))


def enabled() -> bool:
    """Kill switch, read per call so tests/deploys can flip it late."""
    return os.environ.get("TORCHDOCS_RERANK", "1") not in ("0", "false", "no")


_MODEL_LOCK = threading.Lock()
_MODEL = None


def _model():
    # same double-checked load as index/embed._model: exactly one instance,
    # concurrent first calls must not download/build the model twice
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import CrossEncoder

                print(f"[rerank] loading {RERANK_MODEL} (first run downloads ~90MB)")
                _MODEL = CrossEncoder(RERANK_MODEL, device="cpu")
    return _MODEL


def _score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    return [float(s) for s in _model().predict(pairs, show_progress_bar=False)]


def rerank_text(pointer: dict[str, Any], glosses: dict[str, str]) -> str:
    """The candidate line the cross-encoder judges the query against."""
    from index.embed import symbol_from_url

    parts = [
        p
        for p in (
            symbol_from_url(pointer.get("url", "")),
            pointer.get("page_title", ""),
            pointer.get("heading_path", ""),
            glosses.get(pointer.get("url", ""), ""),
        )
        if p
    ]
    return ". ".join(parts)


def rerank(
    query: str,
    pointers: list[dict[str, Any]],
    k: int,
    scorer=None,
) -> list[dict[str, Any]]:
    """Reorder candidate pointers by cross-encoder relevance; return the top-k.

    `scorer` (tests) replaces the model: (list of (query, text) pairs) → scores.
    Fail-open: any scoring error keeps the incoming (RRF-fused) order.
    """
    if len(pointers) <= 1:
        return pointers[:k]
    from index.embed import load_glosses

    glosses = load_glosses()
    pairs = [(query, rerank_text(p, glosses)) for p in pointers]
    try:
        scores = (scorer or _score_pairs)(pairs)
    except Exception as exc:  # noqa: BLE001 — reranking must never break retrieval
        print(f"[rerank] scoring failed ({type(exc).__name__}: {exc}); keeping fused order")
        return pointers[:k]
    order = sorted(range(len(pointers)), key=lambda i: scores[i], reverse=True)
    return [pointers[i] for i in order[:k]]
