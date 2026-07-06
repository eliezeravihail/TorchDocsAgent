"""Translate non-English search queries to English before retrieval.

The docs corpus and the embedding model are English-only, so a query in
another language retrieves noise. This translates just the search query
(not the answer) via the configured LLM; the agent still answers in the
user's language. English input passes through untouched — no LLM call.
"""

from __future__ import annotations

import re

# any character in the Hebrew/Arabic/Cyrillic/CJK/… ranges → not English
_NON_LATIN = re.compile(r"[^\x00-\x7f]")

_TRANSLATE_SYSTEM = (
    "You translate PyTorch documentation search queries into concise English "
    "keyword queries. Reply with ONLY the English query — no quotes, no "
    "explanation. Keep code identifiers (torch.nn.Linear, SGD, ...) verbatim."
)


def looks_english(text: str) -> bool:
    """Cheap heuristic: mostly-ASCII text needs no translation."""
    non_latin = len(_NON_LATIN.findall(text))
    return non_latin <= max(2, len(text) * 0.1)


def translate_to_english(query: str, *, provider: str | None = None, client=None) -> str:
    """Return an English query. English in → same string out (no LLM call)."""
    if looks_english(query):
        return query

    from agent.llm import GenerationError, _raw_completion

    try:
        english = _raw_completion(
            query, system=_TRANSLATE_SYSTEM, provider=provider, client=client
        ).strip()
    except GenerationError as exc:
        print(f"[translate] failed ({exc}); using original query")
        return query
    # a translation should be short; if the model rambled, keep the first line
    return english.splitlines()[0].strip() if english else query
