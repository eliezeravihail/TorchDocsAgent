"""The manual agent loop: the LLM drives the three tools within budgets.

Each step, a planner call returns a JSON action (search_docs / read_page /
ask_source / answer). We execute it, append a short observation, and repeat
until the model declares it can answer or a budget trips — then generate the
grounded Answer from everything accumulated. agent/graph.py is a LangGraph
twin of this loop that shares the same tools, budgets, and planner.
"""

from __future__ import annotations

import ast
import json
import re

from agent.schemas import Answer, Referral

BUDGETS = {"search_docs": 6, "read_page": 2, "ask_source": 1}
MAX_STEPS = sum(BUDGETS.values()) + 2

PLANNER_SYSTEM = (
    "You are planning how to answer a PyTorch question using tools. Each turn, "
    "reply with ONE json object and nothing else:\n"
    '  {"action":"search_docs","query":"english keywords","library":null,"kind":null}\n'
    '  {"action":"read_page","url":"<a url from a previous search result>"}\n'
    '  {"action":"ask_source","question":"..."}  (ONLY for source-code internals)\n'
    '  {"action":"answer"}  (when the gathered context can answer the question)\n'
    'kind picks the content space: "api" = reference pages (use for catalog '
    'questions like "what loss functions exist?" or to find a specific class), '
    '"tutorial" or "guide" = walkthroughs; null = everything. If a search '
    "returned only tutorials but you need the actual API, search again with "
    'kind "api" and the likely symbol names as the query.\n'
    "Guidance: decompose multi-part questions into several search_docs calls "
    "with different queries. Use read_page to open a promising page in full. "
    "Use ask_source only when the answer needs implementation details the docs "
    "don't cover. Answer as soon as you have enough — don't waste tool calls."
)


def _plan(question: str, transcript: list[str], budgets: dict, provider, client) -> dict:
    from agent.llm import GenerationError, _raw_completion

    state = "\n".join(transcript) if transcript else "(no tools used yet)"
    remaining = ", ".join(f"{t}:{n}" for t, n in budgets.items())
    prompt = (
        f"Question: {question}\n\nTool calls so far:\n{state}\n\n"
        f"Remaining tool budget: {remaining}\nYour next action as one json object:"
    )
    try:
        raw = _raw_completion(prompt, system=PLANNER_SYSTEM, provider=provider, client=client)
    except GenerationError:
        return {"action": "answer"}  # planner unreachable → answer with what we have
    return _parse_action(raw)


def _parse_action(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {"action": "answer"}
    try:
        action = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"action": "answer"}
    if action.get("action") not in BUDGETS and action.get("action") != "answer":
        return {"action": "answer"}
    return action


def _humanize(line: str) -> str:
    """Turn a transcript step into a short grey trace line for the web UI.

    The transcript entries are terse function-call records (see tools_exec.py);
    this renders them as the model's visible reasoning. Falls back to the raw
    line on any shape it doesn't recognise, so a format change degrades to a
    still-useful trace rather than crashing the answer path.
    """
    m = re.match(r"search_docs\((.+?)\) → (\[.*\])$", line)
    if m:
        try:
            query = ast.literal_eval(m.group(1))
            titles = [str(t).split(" > ")[-1] for t in ast.literal_eval(m.group(2))]
        except (ValueError, SyntaxError):
            return f"🔍 {line}"
        shown = " · ".join(t for t in titles[:4] if t)
        return f"🔍 searched “{query}”" + (f" → {shown}" if shown else "")
    if line.startswith("read_page"):
        return "📖 read a full page"
    if line.startswith("ask_source"):
        return "🔗 checked the source-code references"
    return f"• {line}"


def answer_agentic(
    question: str, provider: str | None = None, client=None, progress=None
) -> Answer:
    """Run the tool loop and return a grounded Answer."""
    from agent.grounded import answer_from_sections
    from agent.tools_exec import do_search, execute_tool

    budgets = dict(BUDGETS)
    sections: list[dict] = []
    referrals: list[Referral] = []
    seen_urls: set[str] = set()
    transcript: list[str] = []

    def emit_last():  # surface the step we just recorded, in the UI's grey trace
        if progress and transcript:
            progress(_humanize(transcript[-1]))

    # always retrieve once for the raw question — never depend on a (possibly
    # rate-limited) planner call just to do the obvious first search
    budgets["search_docs"] -= 1
    do_search(question, None, sections=sections, seen_urls=seen_urls, transcript=transcript)
    emit_last()

    for _ in range(MAX_STEPS):
        action = _plan(question, transcript, budgets, provider, client)
        name = action.get("action")

        if name == "answer" or all(v == 0 for v in budgets.values()):
            break
        if budgets.get(name, 0) == 0:
            transcript.append(f"{name}: budget exhausted, pick another action or answer")
            continue
        budgets[name] -= 1

        execute_tool(
            name, action, question,
            sections=sections, referrals=referrals,
            seen_urls=seen_urls, transcript=transcript,
        )
        emit_last()

    if progress:
        progress("✍️ writing the answer")
    return answer_from_sections(
        question, sections, referrals=referrals, provider=provider, client=client
    )
