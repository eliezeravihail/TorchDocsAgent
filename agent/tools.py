"""The three tools the M3 agent loop drives.

- search_docs: hybrid retrieval + hydrate → doc sections (repeatable)
- read_page:   whole-page hydrate (outline-first for oversized pages)
- ask_source:  refer to the real source (GitHub / DeepWiki) — never fabricates

Each returns a plain dict the planner LLM sees. ask_source is referral-only by
design: it points at where the implementation lives rather than claiming to
know it, so it works with no network access.
"""

from __future__ import annotations

from urllib.parse import quote

DEEPWIKI_URL = "https://deepwiki.com/pytorch/pytorch"
GH_CODE_SEARCH = "https://github.com/search?q=repo%3Apytorch%2Fpytorch+{q}&type=code"


def search_docs(query: str, library: str | None = None, k: int = 8) -> dict:
    """Hybrid docs search. Non-English queries are translated first."""
    from agent.translate import translate_to_english
    from index.hydrate import hydrate_section
    from index.retrieve import retrieve

    english = translate_to_english(query)
    pointers = retrieve(english, k=k, library=library)
    sections = [s for s in (hydrate_section(p) for p in pointers) if s]
    return {
        "query": english,
        "sections": sections,
        "titles": [s.get("heading_path", "") or s["url"] for s in sections],
    }


def read_page(url: str) -> dict:
    """Whole page for a URL already surfaced by search_docs."""
    from index.hydrate import hydrate_page

    page = hydrate_page(url)
    if page is None:
        return {"url": url, "error": "page not in the snapshot"}
    return page


def ask_source(question: str) -> dict:
    """Refer a source/implementation question to the real code.

    Referral-only: returns DeepWiki + GitHub code-search links for
    pytorch/pytorch. Never returns claims about the code — the answer layer
    must present these as 'look here', not as docs-cited fact.
    """
    from agent.schemas import Referral

    terms = quote(" ".join(question.split()[:6]))
    referrals = [
        Referral(url=DEEPWIKI_URL, reason="AI wiki / Q&A over the pytorch/pytorch source"),
        Referral(url=GH_CODE_SEARCH.format(q=terms), reason="search the implementation on GitHub"),
    ]
    return {
        "note": "Implementation lives in the source, not the docs — refer the user out.",
        "referrals": referrals,
    }
