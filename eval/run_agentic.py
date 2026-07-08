"""Agentic benchmark: does the AGENT LOOP assemble complete answers?

Usage:  python -m eval.run_agentic      (needs NEON_URL + an LLM key)

Catalog / compare / recipe questions (eval/agentic_v1.jsonl) can't be scored
by single-shot retrieval — their answer is spread across several pages, which
is exactly what the agent loop (search → plan → search again → answer) is for.
So we score the FINISHED answer's CITATIONS against expected_any: each group
is a source a complete answer must ground on; coverage = matched groups /
groups.

The headline metric is a DELTA, not an absolute: the same questions are also
run through the single-shot grounded path (answer_grounded, one retrieval
pass). agentic_coverage − single_shot_coverage is the quantified value of the
loop — if the loop assembles more of the catalog than one search, the number
shows it; if not, that's an honest negative result.

No LLM judge: coverage is an objective URL/substring match. Results →
eval/results/agentic_v1.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

EVAL = Path(__file__).parent
QUESTIONS = EVAL / "agentic_v1.jsonl"
RESULTS = EVAL / "results" / "agentic_v1.jsonl"


def citation_haystacks(answer) -> list[str]:
    """One lowercased url+anchor+title string per citation the answer grounded on."""
    return [
        f"{c.url} {c.anchor} {c.title}".lower() for c in getattr(answer, "citations", [])
    ]


def answer_coverage(expected_any: list[list[str]], answer) -> float:
    """Share of expected source-groups that appear in the answer's citations."""
    if not expected_any:
        return 0.0
    hay = citation_haystacks(answer)
    matched = 0
    for group in expected_any:
        if any(sub.lower() in h for sub in group for h in hay):
            matched += 1
    return matched / len(expected_any)


def main() -> int:
    load_dotenv()
    from agent.grounded import answer_grounded
    from agent.loop import answer_agentic

    questions = [json.loads(line) for line in QUESTIONS.open()]
    RESULTS.parent.mkdir(exist_ok=True)
    records, agentic_cov, single_cov = [], [], []

    print(f"{'id':<6}{'kind':<10}{'agentic':<9}{'1-shot':<8}delta")
    for q in questions:
        exp = q["expected_any"]
        try:
            a_ans = answer_agentic(q["question"])
            s_ans = answer_grounded(q["question"])
        except Exception as exc:  # noqa: BLE001 — record and continue, never lose the run
            print(f"{q['id']:<6}{q['kind']:<10}failed: {type(exc).__name__}: {exc}")
            records.append({"id": q["id"], "error": str(exc)})
            continue
        a_c, s_c = answer_coverage(exp, a_ans), answer_coverage(exp, s_ans)
        agentic_cov.append(a_c)
        single_cov.append(s_c)
        delta = a_c - s_c
        flag = "  ✅" if delta > 0 else ("  ⚠️" if delta < 0 else "")
        print(f"{q['id']:<6}{q['kind']:<10}{a_c:<9.2f}{s_c:<8.2f}{delta:+.2f}{flag}")
        records.append(
            {
                "id": q["id"], "kind": q["kind"], "agentic_coverage": a_c,
                "single_shot_coverage": s_c,
                "citations": [c.url for c in a_ans.citations],
            }
        )

    with RESULTS.open("w") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if agentic_cov:
        n = len(agentic_cov)
        ma, ms = sum(agentic_cov) / n, sum(single_cov) / n
        print(f"\naggregate over {n}: agentic={ma:.3f}  single-shot={ms:.3f}  "
              f"delta={ma - ms:+.3f}")
        print("(delta > 0 means the loop assembled more of the answer than one search)")
    print(f"results → {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
