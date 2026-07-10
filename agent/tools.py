"""The three tools the agent loop drives.

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

# dropped from the GitHub code-search query: they add no signal and crowd out
# the discriminating terms (a code-search URL has a practical length limit)
_STOPWORDS = frozenset(
    "a an the is are was were be to of in on for how do does did i you it "
    "what when where why which that this with and or from can could should "
    "my me use using implement implemented implementation work works".split()
)
_MAX_SEARCH_TERMS = 12  # keep the URL sane while retaining the meaningful words


def _search_terms(question: str) -> str:
    """Meaningful words from the question, URL-encoded for GitHub code search.

    Drops stopwords (keeping code identifiers like torch.nn.Linear verbatim)
    and caps the count so the URL stays sane — without throwing away the words
    that discriminate the query (the old code kept only the first 6 words,
    which dropped the actual subject of longer questions).
    """
    words = [w for w in question.split() if w.lower().strip("?.,:;()") not in _STOPWORDS]
    kept = (words or question.split())[:_MAX_SEARCH_TERMS]
    return quote(" ".join(kept))


# content spaces the planner may restrict a search to (must match ingest's
# page_kind values); anything else from the model is ignored, not an error
SEARCH_KINDS = frozenset({"api", "tutorial", "guide"})


def search_docs(
    query: str, library: str | None = None, kind: str | None = None, k: int = 8
) -> dict:
    """Hybrid docs search. Non-English queries are translated first.

    `kind` lets the planner choose the content space: 'api' searches only the
    reference pages (catalog questions — "what loss functions exist?"),
    'tutorial'/'guide' only the walkthroughs. Unknown values degrade to an
    unrestricted search rather than failing the tool call.
    """
    from agent.translate import translate_to_english
    from index.hydrate import hydrate_sections
    from index.retrieve import retrieve

    if kind is not None and kind not in SEARCH_KINDS:
        print(f"[search_docs] ignoring unknown kind {kind!r}", flush=True)
        kind = None
    english = translate_to_english(query)
    pointers = retrieve(english, k=k, library=library, kind=kind)
    sections = hydrate_sections(pointers)  # concurrent — each is a live fetch on the Space
    print(
        f"[search_docs] {english!r} (kind={kind}) → {len(pointers)} pointers, "
        f"{len(sections)} hydrated",
        flush=True,
    )
    return {
        "query": english,
        "sections": sections,
        "titles": [s.get("heading_path", "") or s["url"] for s in sections],
    }


def read_page(url: str) -> dict:
    """Whole page for a URL already surfaced by search_docs."""
    from index.hydrate import hydrate_page

    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        # the planner sometimes passes a section HEADING it saw in a search
        # result (e.g. "Build the Neural Network > Define the Class") instead of
        # the url. Don't fetch that (it 'No scheme supplied'-errors and wastes a
        # call) — tell the model exactly what read_page needs so it self-corrects.
        return {
            "url": url,
            "error": "read_page needs the full https:// URL from a search_docs "
            "result's `url` field, not a section title.",
        }
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

    terms = _search_terms(question)
    referrals = [
        Referral(url=DEEPWIKI_URL, reason="AI wiki / Q&A over the pytorch/pytorch source"),
        Referral(url=GH_CODE_SEARCH.format(q=terms), reason="search the implementation on GitHub"),
    ]
    return {
        "note": "Implementation lives in the source, not the docs — refer the user out.",
        "referrals": referrals,
    }
