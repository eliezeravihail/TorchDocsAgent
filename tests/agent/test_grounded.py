from types import SimpleNamespace

from agent.grounded import (
    answer_from_sections,
    answer_grounded,
    build_context,
    validate_citations,
)
from agent.schemas import Answer, Citation

SECTIONS = [
    {
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html",
        "anchor": "torch-optim-sgd",
        "heading_path": "torch.optim.SGD",
        "content": "Implements stochastic gradient descent. lr (float) required.",
    },
    {
        "url": "https://docs.pytorch.org/docs/stable/optim.html",
        "anchor": "how-to-adjust-learning-rate",
        "heading_path": "How to adjust learning rate",
        "content": "torch.optim.lr_scheduler provides several methods.",
    },
]


def test_build_context_numbers_sections_with_urls():
    context = build_context(SECTIONS)
    assert "[1] TITLE: torch.optim.SGD" in context
    assert "[2] TITLE: How to adjust learning rate" in context
    assert SECTIONS[0]["url"] in context


def test_validate_citations_drops_fabricated_urls():
    answer = Answer(
        answer_md="x",
        citations=[
            Citation(url=SECTIONS[0]["url"], anchor="torch-optim-sgd"),
            Citation(url="https://docs.pytorch.org/docs/stable/fake.html"),
        ],
    )
    validated = validate_citations(answer, SECTIONS)
    assert len(validated.citations) == 1
    assert validated.citations[0].url == SECTIONS[0]["url"]


def _fake_llm_client(payload):
    block = SimpleNamespace(type="tool_use", name="submit_answer", input=payload, id="t1")
    response = SimpleNamespace(content=[block])
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=lambda **kw: response)
    return client


def test_answer_grounded_injects_context_and_validates():
    payload = {
        "answer_md": "Use torch.optim.SGD with lr.",
        "symbols_used": ["torch.optim.SGD"],
        "torch_version": "2.12",
        "citations": [{"url": SECTIONS[0]["url"], "anchor": "torch-optim-sgd", "title": ""}],
        "referrals": [],
    }
    answer = answer_grounded(
        "how do I use SGD?",
        provider="anthropic",
        client=_fake_llm_client(payload),
        retrieve_fn=lambda q, k=8: [dict(s) for s in SECTIONS],
        hydrate_fn=lambda p: p if "content" in p else None,
    )
    assert answer.citations and answer.citations[0].anchor == "torch-optim-sgd"


def test_answer_grounded_empty_index_refers_out():
    answer = answer_grounded(
        "anything",
        retrieve_fn=lambda q, k=8: [],
        hydrate_fn=lambda p: None,
    )
    assert "could not find" in answer.answer_md
    assert answer.referrals and "search" in answer.referrals[0].url


def _scripted_anthropic_client(payloads):
    """A fake client whose messages.create returns each payload in turn."""
    responses = iter(
        SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="submit_answer", input=p, id="t1")]
        )
        for p in payloads
    )
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=lambda **kw: next(responses))
    return client


def test_answer_from_sections_regenerates_on_failed_static_check():
    # first answer lists a symbol it never mentions → the `symbols` check fails
    # → one repair round → the cleaner regenerated answer is kept
    bad = {"answer_md": "Use it.", "symbols_used": ["torch.optim.SGD"],
           "torch_version": "2.12", "citations": [], "referrals": []}
    good = {"answer_md": "Use torch.optim.SGD.", "symbols_used": ["torch.optim.SGD"],
            "torch_version": "2.12", "citations": [], "referrals": []}
    answer = answer_from_sections(
        "how do I use SGD?",
        [dict(s) for s in SECTIONS],
        provider="anthropic",
        client=_scripted_anthropic_client([bad, good]),
    )
    assert "torch.optim.SGD" in answer.answer_md
    assert answer.warning == ""  # repair succeeded → no degraded flag


def test_answer_from_sections_flags_answer_that_still_fails_check():
    # both the first answer and the repair list a symbol they never mention →
    # the check keeps failing → the answer ships (never blocked) but must carry
    # a visible warning so the degradation isn't silent
    bad = {"answer_md": "Use it.", "symbols_used": ["torch.optim.SGD"],
           "torch_version": "2.12", "citations": [], "referrals": []}
    still_bad = {"answer_md": "Just use it.", "symbols_used": ["torch.optim.SGD"],
                 "torch_version": "2.12", "citations": [], "referrals": []}
    answer = answer_from_sections(
        "how do I use SGD?",
        [dict(s) for s in SECTIONS],
        provider="anthropic",
        client=_scripted_anthropic_client([bad, still_bad]),
    )
    assert answer.warning  # non-empty
    assert "symbols" in answer.warning
