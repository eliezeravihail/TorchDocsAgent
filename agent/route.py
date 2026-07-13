"""Route each question to the cheapest path that can answer it well.

Why: the agent loop costs ~5-13 LLM calls per question (planner rounds + the
answer + a possible repair). On the deployment's models that is minutes of
wall-clock — unacceptable for the common case. But the loop's value is real
only for MULTI-SOURCE questions (catalog / compare / recipe / internals),
where it assembles pages one search can't: measured agentic coverage 0.567 vs
0.133 single-shot. A usage question ("how do I use SGD with momentum?") is
answered by ONE retrieval pass + ONE generation — backed by the hybrid
per-kind RRF retrieval (recall@8 ~0.81 on real content).

So: a zero-LLM-call heuristic sends multi-source shapes to the loop and
everything else to the grounded single-shot path. The heuristic is allowed to
be imperfect because it fails open in both directions — a misrouted simple
question still gets a correct (slower) loop answer, and a grounded answer
that ends up with NO citations escalates to the loop instead of shipping an
unsourced reply.

The guard has already bounced non-English input, so the routing text is the
question as received — the patterns stay English-only.

`progress` (optional) is a sink for short human-readable trace lines
("🔍 searched …", "✍️ writing the answer"); the web UI streams them in grey
while the answer is being assembled. It defaults to None (no-op) so scripts
and tests are unaffected.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agent.schemas import Answer

# multi-source question shapes — the loop's home turf. Kept deliberately
# coarse: these strings decide COST (loop vs single shot), not correctness.
_LOOP_PATTERNS = (
    # catalog: "what/which optimizers exist / are there / are supported / can I use"
    r"\b(what|which)\b.{0,60}\b(exist|are there|are available|are supported|"
    r"does .{0,30}(support|offer|provide)|can i (use|choose))\b",
    r"\blist (all|the|every)\b",
    r"\ball (the|of the|available)\b.{0,40}\b(options|kinds|types|ways|functions|"
    r"losses|optimizers|schedulers|layers|transforms)\b",
    # compare: "difference between X and Y", "X vs Y", "should I use X or Y"
    r"\b(difference|differences) between\b",
    r"\bvs\.?\b|\bversus\b|\bcompared? (to|with)\b|\bpros and cons\b|\btrade-?offs?\b",
    r"\bshould i (use|pick|choose)\b.{0,40}\bor\b",
    # recipe: assemble a working thing end-to-end from several pages
    r"\b(build|create|write|design|make|train)\b.{0,50}\b(model|network|cnn|rnn|"
    r"transformer|classifier|detector|gan|autoencoder|pipeline)\b",
    r"\bend[- ]to[- ]end\b|\bfrom scratch\b|\bstep[- ]by[- ]step\b",
    # internals: the ask_source path lives in the loop
    r"\bhow (is|are)\b.{0,40}\bimplemented\b",
    r"\bsource code\b|\bunder the hood\b|\binternals\b",
)
_LOOP_RE = re.compile("|".join(f"(?:{p})" for p in _LOOP_PATTERNS), re.IGNORECASE)


def needs_loop(english_question: str) -> bool:
    """True when the question's shape calls for multi-source assembly."""
    return bool(_LOOP_RE.search(english_question))


def answer_routed(
    question: str,
    provider: str | None = None,
    client=None,
    progress: Callable[[str], None] | None = None,
) -> Answer:
    """Answer via the cheapest adequate path; escalate when grounding fails.

    The guard (app.main / scripts.ask) has already bounced non-English input, so
    the question is English here — no translation step.
    """
    if needs_loop(question):
        from agent.loop import answer_agentic

        return answer_agentic(question, provider=provider, client=client, progress=progress)

    from agent.grounded import answer_grounded

    answer = answer_grounded(question, provider=provider, client=client, progress=progress)
    if answer.citations:
        return answer

    # nothing grounded a simple-looking question — the loop can reformulate
    # and search again, which one fixed pass cannot
    print("[route] grounded answer had no citations; escalating to the loop", flush=True)
    if progress:
        progress("↻ no sources yet — searching more thoroughly")
    from agent.loop import answer_agentic

    return answer_agentic(question, provider=provider, client=client, progress=progress)
