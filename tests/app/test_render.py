import pytest

from agent.schemas import Answer, Citation, Referral
from app.main import THINKING_NOTE, _pipeline, render, respond


@pytest.fixture(autouse=True)
def _disable_guard(monkeypatch):
    # the guard is exercised in tests/agent/test_guard.py; here it would try to
    # load a real model / hit the DB, so switch it off for the render/respond tests
    monkeypatch.setenv("TORCHDOCS_GUARD", "0")


def test_render_includes_answer_citations_referrals():
    answer = Answer(
        answer_md="Use `torch.optim.SGD`.",
        torch_version="2.12",
        citations=[
            Citation(
                url="https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html",
                anchor="torch.optim.SGD",
                title="torch.optim.SGD",
            )
        ],
        referrals=[Referral(url="https://deepwiki.com/pytorch/pytorch", reason="source")],
    )
    md = render(answer)
    assert "Use `torch.optim.SGD`." in md
    assert "**Sources**" in md
    assert "torch.optim.SGD.html#torch.optim.SGD" in md
    assert "**Beyond these docs**" in md
    assert "PyTorch 2.12" in md
    # license link under the citations: text is the license name
    assert "[BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE)" in md


def test_render_no_license_note_without_citations():
    # an answer that quoted nothing (empty index) shouldn't claim a source license
    answer = Answer(answer_md="I could not find anything.", torch_version="unknown")
    assert "BSD-3-Clause" not in render(answer)


def test_pipeline_empty_question():
    assert "Ask me something" in _pipeline("   ")


def test_pipeline_never_crashes_and_never_leaks_the_error(monkeypatch):
    def boom(q, **k):
        raise RuntimeError("db-host.internal:5432 down")

    monkeypatch.setattr("app.main.answer_routed", boom)
    out = _pipeline("how do I use SGD?")
    # the user gets a generic line; the exception text (hosts, slugs, config)
    # goes to the logs only
    assert "went wrong" in out
    assert "db-host.internal" not in out


def test_respond_streams_the_thinking_note_then_the_answer(monkeypatch):
    # the UI generator shows immediate feedback, then swaps in the final answer;
    # the LAST value is what gradio_client (the smoke test) receives
    monkeypatch.setattr(
        "app.main.answer_routed", lambda q, **k: Answer(answer_md="the answer")
    )
    chunks = list(respond("how do I use SGD?"))
    assert chunks[0] == THINKING_NOTE
    assert "the answer" in chunks[-1] and chunks[-1] != THINKING_NOTE


def _cited_answer(text):
    return Answer(
        answer_md=text,
        citations=[
            Citation(
                url="https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html",
                anchor="sgd",
                title="torch.optim.SGD",
            )
        ],
    )


def test_respond_regenerates_when_the_cited_docs_drifted(monkeypatch):
    # stale-while-revalidate: the answer ships first; when the freshness pass
    # reports the cited page drifted, a REGENERATED answer is swapped in with
    # the note — the last yielded value is what the user ends up seeing
    from app.main import FRESHNESS_NOTE

    calls = {"n": 0}

    def routed(q, **k):
        calls["n"] += 1
        return _cited_answer("stale answer" if calls["n"] == 1 else "fresh answer")

    monkeypatch.setattr("app.main.answer_routed", routed)
    monkeypatch.setattr("index.freshness.refresh_pages", lambda urls: set(urls))
    chunks = list(respond("how do I use SGD?"))
    assert "fresh answer" in chunks[-1] and FRESHNESS_NOTE in chunks[-1]
    assert calls["n"] == 2  # answered once, regenerated once


def test_respond_freshness_is_silent_when_nothing_drifted(monkeypatch):
    monkeypatch.setattr("app.main.answer_routed", lambda q, **k: _cited_answer("the answer"))
    monkeypatch.setattr("index.freshness.refresh_pages", lambda urls: set())
    chunks = list(respond("how do I use SGD?"))
    assert "the answer" in chunks[-1] and "regenerated" not in chunks[-1]


def test_respond_freshness_failure_never_disturbs_the_answer(monkeypatch):
    monkeypatch.setattr("app.main.answer_routed", lambda q, **k: _cited_answer("the answer"))

    def boom(urls):
        raise RuntimeError("neon hiccup")

    monkeypatch.setattr("index.freshness.refresh_pages", boom)
    chunks = list(respond("how do I use SGD?"))
    assert "the answer" in chunks[-1]  # the shown answer stands


def test_respond_animates_the_wait_so_it_never_looks_frozen(monkeypatch):
    # a multi-second answer must show MOVING feedback, not a single frozen line:
    # between the note and the answer the generator emits animated spinner frames
    import time

    from app.main import THINKING_SPINNER

    monkeypatch.setattr("app.main.THINKING_TICK", 0.02)  # tick fast so frames land quickly

    def slow(q, **k):
        time.sleep(0.15)  # several ticks → several frames while "drafting"
        return Answer(answer_md="the answer")

    monkeypatch.setattr("app.main.answer_routed", slow)
    chunks = list(respond("how do I use SGD?"))
    assert chunks[0] == THINKING_NOTE and "the answer" in chunks[-1]
    frames = chunks[1:-1]
    assert frames  # the wait produced animation, not a frozen note
    assert any(any(c in f for c in THINKING_SPINNER) for f in frames)
