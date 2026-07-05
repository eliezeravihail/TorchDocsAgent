"""Run the v0 question set through answer_question and the static checks.

Usage:  python -m eval.run_v0
Writes one result line per question to eval/results/v0.jsonl and prints
the pass/fail table. Requires ANTHROPIC_API_KEY (see .env.example).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.llm import GenerationError, answer_question
from eval.checks import format_table, run_checks

QUESTIONS = Path(__file__).parent / "questions_v0.jsonl"
RESULTS = Path(__file__).parent / "results" / "v0.jsonl"


def _load_existing() -> dict[str, dict]:
    """Previous results keyed by question id — only ones that got an answer.

    Free-tier daily quotas are scarce; a rerun must never re-spend them on
    questions that already succeeded (or overwrite good data with errors).
    Pass --fresh to discard previous results and re-run everything.
    """
    if "--fresh" in sys.argv or not RESULTS.exists():
        return {}
    records = (json.loads(line) for line in RESULTS.open())
    return {r["id"]: r for r in records if "answer" in r}


def main() -> int:
    load_dotenv()
    RESULTS.parent.mkdir(exist_ok=True)
    existing = _load_existing()

    rows = []
    records = []
    for line in QUESTIONS.open():
        q = json.loads(line)
        if q["id"] in existing:
            record = existing[q["id"]]
            print(f"[{q['id']}] kept from previous run", flush=True)
        else:
            print(f"[{q['id']}] {q['question'][:70]}...", flush=True)
            record = {"id": q["id"], "type": q["type"], "question": q["question"]}
            try:
                answer = answer_question(q["question"])
                record["answer"] = answer.model_dump()
                record["checks"] = run_checks(answer)
            except GenerationError as exc:
                print(f"  generation failed: {exc}")
                record["error"] = str(exc)
        records.append(record)
        if "checks" in record:
            rows.append((record["id"], record["checks"]))

    with RESULTS.open("w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if rows:
        print("\n" + format_table(rows))
        passed = sum(1 for _, r in rows if all(v is None for v in r.values()))
        print(f"\n{passed}/{len(rows)} answers passed all checks → {RESULTS}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
