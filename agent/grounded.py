"""Grounded answering: retrieve → hydrate → answer with citations (M2 wiring).

This is the single-shot version — one retrieval pass per question. The M3
agent loop replaces the fixed pass with tool calls, but the grounding
contract (context-only answers, exact citations, honest referrals) is
identical.
"""

from __future__ import annotations

from agent.llm import answer_question
from agent.schemas import Answer, Referral

GROUNDED_SYSTEM = (
    "You are a PyTorch documentation assistant. Answer ONLY from the numbered "
    "context sections provided; do not use outside knowledge for claims. "
    "Fill `citations` with the url/anchor/title of every section you used, "
    "copied EXACTLY as given. If the context does not cover part of the "
    "question, say so plainly in the answer and add a `referral` instead of "
    "guessing. List every PyTorch symbol you mention in symbols_used, set "
    "torch_version to the version the context documents, and write clear "
    "markdown with short illustrative snippets where helpful."
)

SECTION_CHAR_LIMIT = 2500
SEARCH_URL = "https://docs.pytorch.org/docs/stable/search.html?q="


def build_context(sections: list[dict]) -> str:
    blocks = []
    for i, section in enumerate(sections, start=1):
        blocks.append(
            f"[{i}] TITLE: {section.get('heading_path', '')}\n"
            f"URL: {section['url']}\n"
            f"ANCHOR: {section.get('anchor', '')}\n"
            f"{section['content'][:SECTION_CHAR_LIMIT]}"
        )
    return "\n\n---\n\n".join(blocks)


def validate_citations(answer: Answer, sections: list[dict]) -> Answer:
    """Keep only citations that point at sections we actually provided."""
    allowed = {(s["url"], s.get("anchor", "")) for s in sections}
    allowed_urls = {s["url"] for s in sections}
    kept = [
        c
        for c in answer.citations
        if (c.url, c.anchor) in allowed or c.url in allowed_urls
    ]
    dropped = len(answer.citations) - len(kept)
    if dropped:
        print(f"[grounded] dropped {dropped} citation(s) not in the provided context")
    return answer.model_copy(update={"citations": kept})


def answer_grounded(
    question: str,
    k: int = 8,
    provider: str | None = None,
    client=None,
    retrieve_fn=None,
    hydrate_fn=None,
) -> Answer:
    """One retrieval pass → grounded answer with validated citations."""
    if retrieve_fn is None:
        from index.retrieve import retrieve as retrieve_fn
    if hydrate_fn is None:
        from index.hydrate import hydrate_section as hydrate_fn

    pointers = retrieve_fn(question, k=k)
    sections = [s for s in (hydrate_fn(p) for p in pointers) if s]

    if not sections:
        return Answer(
            answer_md=(
                "I could not find anything in the PyTorch documentation index "
                "for this question."
            ),
            referrals=[
                Referral(url=SEARCH_URL + question.replace(" ", "+"), reason="docs search")
            ],
        )

    user = f"{build_context(sections)}\n\n---\n\nQuestion: {question}"
    answer = answer_question(user, system=GROUNDED_SYSTEM, provider=provider, client=client)
    return validate_citations(answer, sections)
