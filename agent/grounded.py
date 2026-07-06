"""Grounded answering: retrieve → hydrate → answer with citations (M2 wiring).

This is the single-shot version — one retrieval pass per question. The M3
agent loop replaces the fixed pass with tool calls, but the grounding
contract (context-only answers, exact citations, honest referrals) is
identical.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from agent.llm import GenerationError, answer_question
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


def _regenerate_if_checks_fail(
    question: str, user: str, answer: Answer, provider, client
) -> Answer:
    """Run the static checks (parses / imports / symbols); one repair round.

    This is the live-path wiring of eval/checks.py the plan (M3.2) called for:
    if a code block doesn't parse, an import is off-family, or a listed symbol
    is missing from the prose, re-ask once with the reasons. Keep the repair
    only if it is actually cleaner; never block the user on a failed check.
    """
    from eval.checks import run_checks

    failures = {name: msg for name, msg in run_checks(answer).items() if msg}
    if not failures:
        return answer
    reasons = "; ".join(f"{name}: {msg}" for name, msg in failures.items())
    print(f"[grounded] static checks failed ({reasons}); regenerating once", flush=True)
    repair = (
        f"{user}\n\n---\n\nYour previous answer failed these checks: {reasons}. "
        "Fix them: every fenced python block must parse, imports must be "
        "torch-family or stdlib only, and every symbol in symbols_used must "
        "appear verbatim in the answer. Reply again with a corrected answer."
    )
    try:
        regenerated = answer_question(
            repair, system=GROUNDED_SYSTEM, provider=provider, client=client
        )
    except GenerationError:
        return answer  # repair round unreachable → keep the first answer
    if sum(1 for msg in run_checks(regenerated).values() if msg) < len(failures):
        return regenerated
    return answer


def answer_from_sections(
    question: str,
    sections: list[dict],
    referrals: list[Referral] | None = None,
    provider: str | None = None,
    client=None,
) -> Answer:
    """Generate a grounded answer from already-hydrated sections.

    Shared by the single-shot M2 path and the M3 agent loop — both accumulate
    sections (one pass vs. several tool calls) and end here.
    """
    referrals = referrals or []
    if not sections:
        return Answer(
            answer_md=(
                "I could not find anything in the PyTorch documentation index "
                "for this question."
            ),
            referrals=referrals
            or [Referral(url=SEARCH_URL + quote_plus(question), reason="docs search")],
        )

    user = f"{build_context(sections)}\n\n---\n\nQuestion: {question}"
    answer = answer_question(user, system=GROUNDED_SYSTEM, provider=provider, client=client)
    answer = _regenerate_if_checks_fail(question, user, answer, provider, client)
    answer = validate_citations(answer, sections)
    if referrals:  # tool-loop referrals (e.g. ask_source) join any the model added
        answer = answer.model_copy(update={"referrals": answer.referrals + referrals})
    return answer


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
    return answer_from_sections(question, sections, provider=provider, client=client)
