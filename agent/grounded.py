"""Grounded answering: retrieve → hydrate → answer with citations.

This is the single-shot path — one retrieval pass per question. The agent
loop (agent/loop.py) replaces the fixed pass with tool calls, but the
grounding contract (context-only answers, exact citations, honest referrals)
is identical, and both end in answer_from_sections below.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from agent.llm import GenerationError, answer_question
from agent.schemas import Answer, Referral

GROUNDED_SYSTEM = (
    "You are a PyTorch documentation assistant. Answer ONLY from the numbered "
    "context sections provided; do not use outside knowledge for claims. "
    "Be CONCISE: a few short sentences, or 2-3 short steps for a how-to — and "
    "at most ONE small code snippet, only when it genuinely clarifies. Do NOT "
    "write a comprehensive tutorial or restate whole doc pages; the citations "
    "link to the full docs for depth. Lead with the direct answer. "
    "Fill `citations` with the url/anchor/title of every section you used, "
    "copied EXACTLY as given. If the context does not cover part of the "
    "question, say so plainly in the answer and add a `referral` instead of "
    "guessing. List every PyTorch symbol you mention in symbols_used, and set "
    "torch_version to the version the context documents."
)

# Per-section context budget. A whole page can be far larger than one answer
# needs, and every section shares the prompt, so we cap each one — but the cut
# is made VISIBLE (marker + log) so the model can referral out instead of
# silently answering from a truncated view.
SECTION_CHAR_LIMIT = 2500
SEARCH_URL = "https://docs.pytorch.org/docs/stable/search.html?q="


def _section_body(section: dict) -> str:
    content = section.get("content", "")
    if len(content) <= SECTION_CHAR_LIMIT:
        return content
    print(
        f"[grounded] section {section.get('url', '')} truncated for context "
        f"({len(content)} → {SECTION_CHAR_LIMIT} chars)",
        flush=True,
    )
    marker = "\n… [section truncated — see the source URL for the rest]"
    return content[:SECTION_CHAR_LIMIT] + marker


def build_context(sections: list[dict]) -> str:
    blocks = []
    for i, section in enumerate(sections, start=1):
        blocks.append(
            f"[{i}] TITLE: {section.get('heading_path', '')}\n"
            f"URL: {section.get('url', '')}\n"
            f"ANCHOR: {section.get('anchor', '')}\n"
            f"{_section_body(section)}"
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


def _regenerate_if_checks_fail(user: str, answer: Answer, provider, client) -> Answer:
    """Run the static checks (parses / imports / symbols); one repair round.

    This wires eval/checks.py into the live answer path: if a code block
    doesn't parse, an import is off-family, or a listed symbol is missing from
    the prose, re-ask once with the reasons. Keep the repair only if it is
    actually cleaner; never block the user on a failed check.
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
    answer = _regenerate_if_checks_fail(user, answer, provider, client)
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

    pointers = retrieve_fn(question, k=k)
    if hydrate_fn is None:
        # default path: hydrate the k sections CONCURRENTLY — on the Space each
        # is a live page fetch, and doing them in series was the dominant latency
        from index.hydrate import hydrate_sections

        sections = hydrate_sections(pointers)
    else:  # an injected hydrate_fn (tests) stays sequential and deterministic
        sections = [s for s in (hydrate_fn(p) for p in pointers) if s]
    return answer_from_sections(question, sections, provider=provider, client=client)
