"""Retrieval benchmark: does search_docs surface the RIGHT pages?

Usage:  python -m eval.run_retrieval        (needs NEON_URL; questions are English)
        TORCHDOCS_EVAL_SET=v0 python -m eval.run_retrieval   (the old 13-q set)

Measures the retrieval layer alone — no LLM, no answer generation. Each
question carries expected sources: a list of GROUPS, each group a list of
alternative URL/title substrings (any alternative counts as that source
found). Questions with no groups (referral-only edge cases) are skipped.

Default set is v1 (eval/questions_v1.jsonl, 100 questions authored against the
verified external docs inventory — expected sources inline). v0 is the
original 13-question set (questions_v0.jsonl + retrieval_v0.jsonl).

Metrics per question, at k = TORCHDOCS_RETRIEVAL_K (default 8, the app's k):
  recall@k — matched groups / expected groups
  MRR      — 1 / rank of the first pointer matching any group

Run BEFORE and AFTER retrieval-affecting changes; the aggregate line is the
comparison. Results are written to eval/results/retrieval_<set>.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).parent
EVAL_SET = os.environ.get("TORCHDOCS_EVAL_SET", "v1")
_K = os.environ.get("TORCHDOCS_RETRIEVAL_K", "8")


def _results_path(rerank_on: bool) -> Path:
    """Standard file for the reranked run; a _norerank sibling for the ablation
    (a k≠8 sweep suffixes k too) — so neither masquerades as the other."""
    variant = ("" if _K == "8" else f"_k{_K}") + ("" if rerank_on else "_norerank")
    return EVAL_DIR / "results" / f"retrieval_{EVAL_SET}{variant}.jsonl"


def load_questions() -> dict[str, dict]:
    """{id: {question, expected}} for the selected set.

    v1 carries expected sources inline; v0 keeps them in a sibling file.
    """
    if EVAL_SET == "v0":
        qs = (EVAL_DIR / "questions_v0.jsonl").open()
        questions = {json.loads(x)["id"]: json.loads(x) for x in qs}
        for line in (EVAL_DIR / "retrieval_v0.jsonl").open():
            row = json.loads(line)
            questions.get(row["id"], {})["expected"] = row["expected"]
        return questions
    return {
        json.loads(x)["id"]: json.loads(x)
        for x in (EVAL_DIR / f"questions_{EVAL_SET}.jsonl").open()
    }


def pointer_text(pointer: dict) -> str:
    """The haystack a pattern is matched against: url + titles, lowercased."""
    return " ".join(
        [pointer.get("url", ""), pointer.get("page_title", ""), pointer.get("heading_path", "")]
    ).lower()


def group_rank(group: list[str], pointers: list[dict]) -> int | None:
    """1-based rank of the first pointer matching ANY of the group's patterns."""
    for rank, pointer in enumerate(pointers, start=1):
        text = pointer_text(pointer)
        if any(pattern.lower() in text for pattern in group):
            return rank
    return None


def question_metrics(expected: list[list[str]], pointers: list[dict]) -> dict:
    """recall (matched groups / groups) and MRR for one question."""
    ranks = [group_rank(group, pointers) for group in expected]
    hits = [r for r in ranks if r is not None]
    return {
        "recall": len(hits) / len(expected),
        "mrr": 1.0 / min(hits) if hits else 0.0,
        "ranks": ranks,
    }


def _measure(questions: dict[str, dict], k: int, *, rerank_on: bool) -> dict:
    """One retrieval pass over the set at a fixed rerank setting.

    The rerank on/off toggle is TORCHDOCS_RERANK, which retrieve() reads (via
    index.rerank.enabled()) fresh on every call — so flipping the env here runs
    the exact production path with the reranker in or out. Returns the aggregate
    plus per-question records (also written to the pass's own results file)."""
    os.environ["TORCHDOCS_RERANK"] = "1" if rerank_on else "0"
    from index.retrieve import retrieve

    label = "rerank ON " if rerank_on else "rerank OFF"
    records, recalls, mrrs = [], [], []
    print(f"\n── {label} ──  eval set: {EVAL_SET} ({len(questions)} questions), k={k}")
    print(f"{'id':<6}{'recall@' + str(k):<12}{'MRR':<8}misses")
    for qid, q in questions.items():
        expected = q.get("expected", [])
        if not expected:
            print(f"{qid:<6}{'—':<12}{'—':<8}(no docs source expected — skipped)")
            continue
        pointers = retrieve(q["question"], k=k)
        m = question_metrics(expected, pointers)
        recalls.append(m["recall"])
        mrrs.append(m["mrr"])
        misses = [
            "/".join(g) for g, r in zip(expected, m["ranks"], strict=True) if r is None
        ]
        print(f"{qid:<6}{m['recall']:<12.2f}{m['mrr']:<8.2f}{', '.join(misses)}")
        records.append(
            {"id": qid, "question": q["question"], "k": k, **m,
             "retrieved": [p["url"] + "#" + p.get("anchor", "") for p in pointers]}
        )
    path = _results_path(rerank_on)
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    n = len(recalls)
    return {
        "recall": sum(recalls) / n if n else 0.0,
        "mrr": sum(mrrs) / n if n else 0.0,
        "n": n,
        "path": path,
    }


def main() -> int:
    load_dotenv()
    k = int(os.environ.get("TORCHDOCS_RETRIEVAL_K", "8"))
    questions = load_questions()

    # measure BOTH configurations every run — the reranker's contribution is the
    # ablation the team keeps asking for, so make it a permanent side-by-side
    # instead of a manual rerank=0 dispatch nobody can trigger from mobile.
    on = _measure(questions, k, rerank_on=True)
    off = _measure(questions, k, rerank_on=False)
    if not on["n"]:
        print("no questions with expectations — nothing measured")
        return 1

    d_recall = on["recall"] - off["recall"]
    d_mrr = on["mrr"] - off["mrr"]
    print(f"\n{'config':<14}{'recall@' + str(k):<12}{'MRR':<8}")
    print(f"{'rerank ON':<14}{on['recall']:<12.3f}{on['mrr']:<8.3f}")
    print(f"{'rerank OFF':<14}{off['recall']:<12.3f}{off['mrr']:<8.3f}")
    print(f"{'Δ (rerank)':<14}{d_recall:<+12.3f}{d_mrr:<+8.3f}")
    print(f"\nresults → {on['path']}  |  {off['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
