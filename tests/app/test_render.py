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


def test_respond_empty_question():
    assert "Ask me something" in respond("   ")


def test_respond_never_crashes_and_never_leaks_the_error(monkeypatch):
    def boom(q, **k):
        raise RuntimeError("db-host.internal:5432 down")

    monkeypatch.setattr("app.main.answer_agentic", boom)
    out = respond("how do I use SGD?")
    # the user gets a generic line; the exception text (hosts, slugs, config)
    # goes to the logs only
    assert "went wrong" in out
    assert "db-host.internal" not in out
