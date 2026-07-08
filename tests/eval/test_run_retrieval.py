"""The retrieval benchmark's metric functions — pure, no DB."""

import json
from pathlib import Path

from eval.run_retrieval import group_rank, question_metrics

EVAL = Path(__file__).parent.parent.parent / "eval"


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


def _load(name):
    return [json.loads(line) for line in (EVAL / name).open()]


def test_v1_question_sets_are_well_formed():
    valid = _load("questions_v1.jsonl")
    invalid = _load("invalid_v1.jsonl")
    agentic = _load("agentic_v1.jsonl")
    assert len(valid) == 100 and len(invalid) == 100 and len(agentic) == 20
    # unique ids within each set
    for rows in (valid, invalid, agentic):
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))
    for r in valid:
        assert r["question"] and r["expected"]
        for group in r["expected"]:
            assert group and all(isinstance(p, str) and p for p in group)


def test_v1_expected_sources_are_grounded_in_the_docs_inventory():
    # the anti-memory guarantee, enforced in CI: every expected source in the
    # valid + agentic sets must match a page/symbol the docs SITE actually
    # publishes (eval/docs_inventory.jsonl), not something invented.
    inv = _load("docs_inventory.jsonl")
    hay = [(r["url"].lower(), r["name"].lower()) for r in inv]

    def grounded(sub: str) -> bool:
        s = sub.lower()
        return any(s in url or s in name for url, name in hay)

    ungrounded = []
    for name in ("questions_v1.jsonl", "agentic_v1.jsonl"):
        for r in _load(name):
            for group in r.get("expected", []) + r.get("expected_any", []):
                if not any(grounded(sub) for sub in group):
                    ungrounded.append((r["id"], group))
    assert not ungrounded, f"expected sources not in the docs inventory: {ungrounded[:10]}"
