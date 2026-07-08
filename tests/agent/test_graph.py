import pytest

from agent.schemas import Answer

pytest.importorskip("langgraph")


class ScriptedPlanner:
    def __init__(self, actions):
        self._actions = list(actions)

    def __call__(self, prompt, system, provider=None, client=None):
        return self._actions.pop(0)


def test_graph_decomposes_and_answers(monkeypatch):
    from agent.graph import answer_graph

    monkeypatch.setattr(
        "agent.llm._raw_completion",
        ScriptedPlanner([
            '{"action":"search_docs","query":"cnn images"}',
            '{"action":"search_docs","query":"dataloader"}',
            '{"action":"answer"}',
        ]),
    )
    calls = {"n": 0}

    def fake_search(query, library=None, kind=None, k=8):
        calls["n"] += 1
        return {"query": query, "sections": [{"url": f"u{calls['n']}", "anchor": "",
                "heading_path": "H", "content": "t"}], "titles": ["H"]}

    monkeypatch.setattr("agent.tools.search_docs", fake_search)

    captured = {}

    def fake_answer(q, s, referrals=None, provider=None, client=None):
        captured["s"] = s
        return Answer(answer_md="done")

    monkeypatch.setattr("agent.grounded.answer_from_sections", fake_answer)

    result = answer_graph("how do I build a CNN?")
    assert result.answer_md == "done"
    # 1 forced seed search + 2 planner-driven searches (parity with the loop)
    assert calls["n"] == 3
    assert len(captured["s"]) == 3


def test_graph_terminates_on_budget(monkeypatch):
    from agent.graph import answer_graph

    monkeypatch.setattr(
        "agent.llm._raw_completion",
        ScriptedPlanner(['{"action":"search_docs","query":"x"}'] * 30),
    )
    monkeypatch.setattr(
        "agent.tools.search_docs",
        lambda q, library=None, kind=None, k=8: {"query": q, "sections": [], "titles": []},
    )
    monkeypatch.setattr(
        "agent.grounded.answer_from_sections",
        lambda q, s, referrals=None, provider=None, client=None: Answer(answer_md="stopped"),
    )
    assert answer_graph("q").answer_md == "stopped"  # must not recurse forever
