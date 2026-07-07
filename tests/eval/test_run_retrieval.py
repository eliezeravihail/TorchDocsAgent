"""The retrieval benchmark's metric functions — pure, no DB."""

import json
from pathlib import Path

from eval.run_retrieval import EXPECTED, QUESTIONS, group_rank, question_metrics


def _p(url, title=""):
    return {"url": url, "page_title": title, "heading_path": ""}


POINTERS = [
    _p("https://docs.pytorch.org/docs/stable/optim.html"),
    _p("https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html"),
    _p("https://docs.pytorch.org/tutorials/beginner/basics.html", "Learn the Basics"),
]


def test_group_rank_finds_first_match_case_insensitively():
    assert group_rank(["generated/torch.optim.sgd"], POINTERS) == 2
    assert group_rank(["OPTIM.HTML"], POINTERS) == 1
    assert group_rank(["no-such-page"], POINTERS) is None


def test_group_rank_any_alternative_counts():
    assert group_rank(["missing-thing", "learn the basics"], POINTERS) == 3


def test_question_metrics_recall_and_mrr():
    expected = [["generated/torch.optim.sgd"], ["nonexistent"]]
    m = question_metrics(expected, POINTERS)
    assert m["recall"] == 0.5  # one of two groups found
    assert m["mrr"] == 0.5  # first hit at rank 2
    assert m["ranks"] == [2, None]


def test_question_metrics_no_hits():
    m = question_metrics([["nope"]], POINTERS)
    assert m["recall"] == 0.0 and m["mrr"] == 0.0


def test_expectations_align_with_the_question_set():
    question_ids = {json.loads(line)["id"] for line in Path(QUESTIONS).open()}
    rows = [json.loads(line) for line in Path(EXPECTED).open()]
    assert {r["id"] for r in rows} == question_ids  # every question has a row
    measured = [r for r in rows if r["expected"]]
    assert len(measured) >= 10  # enough coverage for the aggregate to mean something
    for row in measured:
        for group in row["expected"]:
            assert group and all(isinstance(p, str) and p for p in group)
