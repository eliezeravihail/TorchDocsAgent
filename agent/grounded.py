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
    "Fill `citations` with the url/anchor/title of every section you used, "
    "copied EXACTLY as given. If the context does not cover part of the "
    "question, say so plainly in the answer and add a `referral` instead of "
    "guessing. List every PyTorch symbol you mention in symbols_used, set "
    "torch_version to the version the context documents, and write clear "
    "markdown with short illustrative snippets where helpful."
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


def _flag_unverified(answer: Answer, failures: dict[str, str]) -> Answer:
    """Attach a user-visible warning to an answer that still fails a check.

    A degraded answer is delivered (never blocked), but the user must see that
    it did not pass verification — otherwise an invented symbol or an
    unparseable snippet ships silently. This is the visible half of the
    static-check contract; the reasons themselves go to the logs, not the UI.
    """
    kinds = ", ".join(sorted(failures))
    notice = (
        f"This answer did not pass an automatic check ({kinds}) and may contain "
        "an unverified code snippet or symbol — double-check it against the "
        "linked documentation."
    )
    return answer.model_copy(update={"warning": notice})


def _regenerate_if_checks_fail(user: str, answer: Answer, provider, client) -> Answer:
    """Run the static checks (parses / imports / symbols); one repair round.

    This wires eval/checks.py into the live answer path: if a code block
    doesn't parse, an import is off-family, or a listed symbol is missing from
    the prose, re-ask once with the reasons. Keep the repair only if it is
    actually cleaner; never block the user on a failed check — but if failures
    remain after the round, flag the delivered answer so the gap is visible.
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
        return _flag_unverified(answer, failures)  # repair unreachable → keep, but flag
    regen_failures = {name: msg for name, msg in run_checks(regenerated).items() if msg}
    if len(regen_failures) < len(failures):
        # repair helped; deliver it — flag only if some failures still remain
        return _flag_unverified(regenerated, regen_failures) if regen_failures else regenerated
    return _flag_unverified(answer, failures)  # repair didn't help → keep first, flag it


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
    if hydrate_fn is None:
        from index.hydrate import hydrate_section as hydrate_fn

    pointers = retrieve_fn(question, k=k)
    sections = [s for s in (hydrate_fn(p) for p in pointers) if s]
    return answer_from_sections(question, sections, provider=provider, client=client)
