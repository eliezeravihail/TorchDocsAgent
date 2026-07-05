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


def main() -> int:
    load_dotenv()
    RESULTS.parent.mkdir(exist_ok=True)

    rows = []
    with QUESTIONS.open() as qf, RESULTS.open("w") as out:
        for line in qf:
            q = json.loads(line)
            print(f"[{q['id']}] {q['question'][:70]}...", flush=True)
            record: dict = {"id": q["id"], "type": q["type"], "question": q["question"]}
            try:
                answer = answer_question(q["question"])
                results = run_checks(answer)
                record["answer"] = answer.model_dump()
                record["checks"] = results
                rows.append((q["id"], results))
            except GenerationError as exc:
                print(f"  generation failed: {exc}")
                record["error"] = str(exc)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if rows:
        print("\n" + format_table(rows))
        passed = sum(1 for _, r in rows if all(v is None for v in r.values()))
        print(f"\n{passed}/{len(rows)} answers passed all checks → {RESULTS}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
