"""The same agent, as a LangGraph state machine.

Identical tools, budgets, planner, and forced seed search as agent/loop.py —
the tool step itself is the shared agent/tools_exec.execute_tool, so the two
drivers cannot drift. Only the control flow differs: an explicit graph instead
of a Python while-loop:

    planner ──(action?)──▶ tools ──▶ planner        (cycle)
        └────(answer / budget spent)────▶ generate ──▶ END

The comparison (LoC, debuggability, latency) lives in
docs/loop-vs-langgraph.md.
"""

from __future__ import annotations

from typing import Any, TypedDict

from agent.loop import BUDGETS, MAX_STEPS, _plan
from agent.schemas import Answer, Referral


class AgentState(TypedDict):
    question: str
    provider: str | None
    client: Any
    budgets: dict
    sections: list
    referrals: list
    seen: set
    transcript: list
    steps: int
    action: dict
    answer: Answer | None


def _planner_node(state: AgentState) -> dict:
    action = _plan(
        state["question"], state["transcript"], state["budgets"],
        state["provider"], state["client"],
    )
    return {"action": action, "steps": state["steps"] + 1}


def _route(state: AgentState) -> str:
    action = state["action"]
    if action.get("action") == "answer":
        return "generate"
    if all(v == 0 for v in state["budgets"].values()) or state["steps"] >= MAX_STEPS:
        return "generate"
    return "tools"


def _tools_node(state: AgentState) -> dict:
    from agent.tools_exec import execute_tool

    name = state["action"].get("action")
    budgets, sections = dict(state["budgets"]), list(state["sections"])
    referrals, seen = list(state["referrals"]), set(state["seen"])
    transcript = list(state["transcript"])

    if budgets.get(name, 0) == 0:
        transcript.append(f"{name}: budget exhausted, answer or pick another action")
        return {"budgets": budgets, "transcript": transcript}
    budgets[name] -= 1

    execute_tool(
        name, state["action"], state["question"],
        sections=sections, referrals=referrals, seen_urls=seen, transcript=transcript,
    )

    return {"budgets": budgets, "sections": sections, "referrals": referrals,
            "seen": seen, "transcript": transcript}


def _generate_node(state: AgentState) -> dict:
    from agent.grounded import answer_from_sections

    answer = answer_from_sections(
        state["question"], state["sections"], referrals=state["referrals"],
        provider=state["provider"], client=state["client"],
    )
    return {"answer": answer}


def build_graph():
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("planner", _planner_node)
    g.add_node("tools", _tools_node)
    g.add_node("generate", _generate_node)
    g.set_entry_point("planner")
    g.add_conditional_edges("planner", _route, {"tools": "tools", "generate": "generate"})
    g.add_edge("tools", "planner")
    g.add_edge("generate", END)
    return g.compile()


def answer_graph(question: str, provider: str | None = None, client=None) -> Answer:
    """Run the LangGraph version; returns the same grounded Answer as the loop."""
    from agent.tools_exec import do_search

    graph = build_graph()

    # same forced seed search as the manual loop — retrieve once for the raw
    # question before the (possibly rate-limited) planner ever runs
    budgets = dict(BUDGETS)
    sections: list[dict] = []
    seen: set[str] = set()
    transcript: list[str] = []
    budgets["search_docs"] -= 1
    do_search(question, None, sections=sections, seen_urls=seen, transcript=transcript)

    initial: AgentState = {
        "question": question, "provider": provider, "client": client,
        "budgets": budgets, "sections": sections, "referrals": [], "seen": seen,
        "transcript": transcript, "steps": 0, "action": {}, "answer": None,
    }
    final = graph.invoke(initial, config={"recursion_limit": 2 * MAX_STEPS + 5})
    if final["answer"] is not None:
        return final["answer"]
    # generate always produces an Answer, so this is pure defensiveness — keep
    # it consistent with grounded's empty path instead of a divergent URL
    from agent.grounded import SEARCH_URL

    return Answer(
        answer_md="(no answer produced)",
        referrals=[Referral(url=SEARCH_URL, reason="docs search")],
    )
