from agent.loop import _parse_action, answer_agentic
from agent.schemas import Answer


def test_parse_action_extracts_json_from_noise():
    assert _parse_action('sure! {"action":"search_docs","query":"sgd"} ')["query"] == "sgd"
    assert _parse_action("no json here")["action"] == "answer"
    assert _parse_action('{"action":"bogus"}')["action"] == "answer"


class ScriptedAgent:
    """Drives the loop: a queue of planner actions, then a final Answer JSON."""

    def __init__(self, plan_actions, final_answer):
        self._plans = list(plan_actions)
        self._final = final_answer

    def plan(self, prompt, system, provider=None, client=None):
        return self._plans.pop(0)

    def final(self, *a, **k):
        return self._final


def test_loop_decomposes_and_answers(monkeypatch):
    # planner asks for two searches, then answers
    scripted = ScriptedAgent(
        ['{"action":"search_docs","query":"cnn image classification"}',
         '{"action":"search_docs","query":"dataloader images"}',
         '{"action":"answer"}'],
        None,
    )
    calls = {"search": 0}

    def fake_search(query, library=None, k=8):
        calls["search"] += 1
        return {"query": query, "sections": [{"url": f"u{calls['search']}", "anchor": "",
                "heading_path": "H", "content": "text"}], "titles": ["H"]}

    monkeypatch.setattr("agent.loop._raw_completion_missing", None, raising=False)
    monkeypatch.setattr("agent.llm._raw_completion", scripted.plan)
    monkeypatch.setattr("agent.tools.search_docs", fake_search)

    captured = {}

    def fake_answer(question, sections, referrals=None, provider=None, client=None):
        captured["sections"] = sections
        captured["referrals"] = referrals
        return Answer(answer_md="done")

    monkeypatch.setattr("agent.grounded.answer_from_sections", fake_answer)

    result = answer_agentic("how do I build a CNN for images?")
    assert result.answer_md == "done"
    assert calls["search"] == 2  # decomposed into two distinct searches
    assert len(captured["sections"]) == 2  # both accumulated


def test_loop_source_question_adds_referrals(monkeypatch):
    scripted = ScriptedAgent(
        ['{"action":"ask_source","question":"conv2d internals"}', '{"action":"answer"}'],
        None,
    )
    monkeypatch.setattr("agent.llm._raw_completion", scripted.plan)

    captured = {}

    def fake_answer(question, sections, referrals=None, provider=None, client=None):
        captured["referrals"] = referrals
        return Answer(answer_md="see source")

    monkeypatch.setattr("agent.grounded.answer_from_sections", fake_answer)
    answer_agentic("how is conv2d implemented?")
    assert captured["referrals"]  # ask_source contributed referral links
    assert any("deepwiki" in r.url for r in captured["referrals"])


def test_loop_stops_at_budget(monkeypatch):
    # planner always wants to search; budget must cap it
    scripted = ScriptedAgent(
        ['{"action":"search_docs","query":"x"}'] * 20, None
    )
    monkeypatch.setattr("agent.llm._raw_completion", scripted.plan)
    monkeypatch.setattr(
        "agent.tools.search_docs",
        lambda q, library=None, k=8: {"query": q, "sections": [], "titles": []},
    )
    captured = {}
    monkeypatch.setattr(
        "agent.grounded.answer_from_sections",
        lambda q, s, referrals=None, provider=None, client=None: captured.setdefault("hit", True)
        or Answer(answer_md="x"),
    )
    answer_agentic("q")  # must terminate (MAX_STEPS / budget), not loop forever
    assert captured["hit"]
