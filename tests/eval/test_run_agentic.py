"""The agentic benchmark's coverage scoring — pure, no LLM, no DB."""

from types import SimpleNamespace

from eval.run_agentic import answer_coverage


def _answer(*urls):
    cites = [SimpleNamespace(url=u, anchor="", title="") for u in urls]
    return SimpleNamespace(citations=cites)


def test_full_coverage_when_every_group_is_cited():
    ans = _answer(
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html",
    )
    assert answer_coverage([["crossentropyloss"], ["nllloss"]], ans) == 1.0


def test_partial_coverage_when_one_group_missing():
    ans = _answer("https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html")
    # only 1 of the 2 catalog items was cited → the loop gathered half
    assert answer_coverage([["crossentropyloss"], ["nllloss"]], ans) == 0.5


def test_zero_coverage_when_nothing_matches():
    ans = _answer("https://docs.pytorch.org/tutorials/beginner/basics/intro.html")
    assert answer_coverage([["crossentropyloss"], ["nllloss"]], ans) == 0.0


def test_alternatives_within_a_group_count():
    ans = _answer("https://docs.pytorch.org/docs/stable/amp.html")
    # either "gradscaler" or "amp" satisfies the group; amp.html matches "amp"
    assert answer_coverage([["gradscaler", "amp"]], ans) == 1.0


def test_no_citations_is_zero_not_crash():
    assert answer_coverage([["crossentropyloss"]], _answer()) == 0.0
