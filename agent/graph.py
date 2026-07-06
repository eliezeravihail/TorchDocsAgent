"""The same agent, as a LangGraph state machine (M3.4).

Identical tools, budgets, and planner as agent/loop.py — only the control flow
is expressed as an explicit graph instead of a Python while-loop:

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
    from agent.tools import ask_source, read_page, search_docs

    name = state["action"].get("action")
    budgets, sections = dict(state["budgets"]), list(state["sections"])
    referrals, seen = list(state["referrals"]), set(state["seen"])
    transcript = list(state["transcript"])

    if budgets.get(name, 0) == 0:
        transcript.append(f"{name}: budget exhausted, answer or pick another action")
        return {"budgets": budgets, "transcript": transcript}
    budgets[name] -= 1

    action = state["action"]
    if name == "search_docs":
        result = search_docs(action.get("query", state["question"]), action.get("library"))
        for s in result["sections"]:
            if s["url"] + s.get("anchor", "") not in seen:
                seen.add(s["url"] + s.get("anchor", ""))
                sections.append(s)
        transcript.append(f"search_docs({result['query']!r}) → {result['titles'][:5]}")
    elif name == "read_page":
        page = read_page(action.get("url", ""))
        if "content" in page:
            sections.append({"url": page["url"], "anchor": "",
                             "heading_path": page.get("title", ""), "content": page["content"]})
            transcript.append(f"read_page({page['url']}) → full page added")
        else:
            transcript.append(f"read_page → {page.get('outline') or page.get('error')}")
    elif name == "ask_source":
        src = ask_source(action.get("question", state["question"]))
        referrals.extend(src["referrals"])
        transcript.append(f"ask_source → {len(src['referrals'])} referral links")

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
    graph = build_graph()
    initial: AgentState = {
        "question": question, "provider": provider, "client": client,
        "budgets": dict(BUDGETS), "sections": [], "referrals": [], "seen": set(),
        "transcript": [], "steps": 0, "action": {}, "answer": None,
    }
    final = graph.invoke(initial, config={"recursion_limit": 2 * MAX_STEPS + 5})
    return final["answer"] or Answer(answer_md="(no answer produced)", referrals=[Referral(
        url="https://docs.pytorch.org/docs/stable/search.html", reason="docs search")])
