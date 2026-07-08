import pytest

from agent.schemas import Answer, Citation, Referral
from app.main import render, respond


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


def test_render_shows_degraded_warning_up_top():
    # a degraded answer (failed a static check) must carry a visible flag, not
    # ship silently — the answer body still renders below it
    answer = Answer(
        answer_md="Use `torch.foo.Bar`.",
        warning="This answer did not pass an automatic check (symbols) and may "
        "contain an unverified code snippet or symbol — double-check it against "
        "the linked documentation.",
    )
    md = render(answer)
    assert "⚠️" in md
    assert "did not pass an automatic check" in md
    assert "Use `torch.foo.Bar`." in md
    # the warning comes before the answer body
    assert md.index("⚠️") < md.index("Use `torch.foo.Bar`.")


def test_render_no_warning_when_clean():
    assert "⚠️" not in render(Answer(answer_md="All good."))


def test_respond_empty_question():
    assert "Ask me something" in respond("   ")


def test_respond_never_crashes_and_never_leaks_the_error(monkeypatch):
    def boom(q, **k):
        raise RuntimeError("db-host.internal:5432 down")

    monkeypatch.setattr("app.main.answer_agentic", boom)
    out = respond("how do I use SGD?")
    # a non-LLM failure → the generic line; the exception text (hosts, slugs,
    # config) goes to the logs only
    assert "went wrong" in out
    assert "db-host.internal" not in out


def test_respond_categorizes_llm_failure(monkeypatch):
    from agent.llm import GenerationError

    def no_provider(q, **k):
        raise GenerationError("all providers failed (model-a, model-b): 429")

    monkeypatch.setattr("app.main.answer_agentic", no_provider)
    out = respond("how do I use SGD?")
    # an LLM-layer failure gets its own message so the smoke test (and the user)
    # can tell it apart from an index/DB failure — but still no raw detail leaks
    assert "temporarily unavailable" in out
    assert "went wrong" not in out
    assert "model-a" not in out
    assert "429" not in out
