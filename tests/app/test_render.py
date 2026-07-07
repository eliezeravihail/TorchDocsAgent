from agent.schemas import Answer, Citation, Referral
from app.main import render, respond


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
    # license/attribution note under the citations
    assert "BSD-3-Clause" in md and "PyTorch Contributors" in md


def test_render_no_license_note_without_citations():
    # an answer that quoted nothing (empty index) shouldn't claim a source license
    answer = Answer(answer_md="I could not find anything.", torch_version="unknown")
    assert "BSD-3-Clause" not in render(answer)


def test_respond_empty_question():
    assert "Ask me something" in respond("   ")


def test_respond_never_crashes(monkeypatch):
    def boom(q, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.main.answer_agentic", boom)
    out = respond("how do I use SGD?")
    assert "went wrong" in out and "db down" in out
