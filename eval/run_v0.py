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

from agent.llm import answer_question
from eval.checks import format_table, run_checks

QUESTIONS = Path(__file__).parent / "questions_v0.jsonl"
GROUNDED = "--grounded" in sys.argv
RESULTS = Path(__file__).parent / "results" / ("v0-grounded.jsonl" if GROUNDED else "v0.jsonl")


def _grounded_api_rate(conn, symbols: list[str]) -> float | None:
    """Share of symbols_used that exist somewhere in the docs index."""
    if not symbols:
        return None
    hits = 0
    for symbol in symbols:
        row = conn.execute(
            "select 1 from chunks where tsv @@ plainto_tsquery('english', %s) limit 1",
            (symbol,),
        ).fetchone()
        hits += 1 if row else 0
    return hits / len(symbols)


def _flush(records: list[dict], path: Path) -> None:
    """Write all results so far. Called after every question so a crash mid-run
    never discards answers already paid for out of the scarce daily quota."""
    with path.open("w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


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

    conn = None
    if GROUNDED:
        from agent.grounded import answer_grounded
        from index.db import connect

        conn = connect()

    rows = []
    records = []
    try:
        with QUESTIONS.open() as questions:
            for line in questions:
                q = json.loads(line)
                if q["id"] in existing:
                    record = existing[q["id"]]
                    print(f"[{q['id']}] kept from previous run", flush=True)
                else:
                    print(f"[{q['id']}] {q['question'][:70]}...", flush=True)
                    record = {"id": q["id"], "type": q["type"], "question": q["question"]}
                    try:
                        if GROUNDED:
                            answer = answer_grounded(q["question"])
                            record["grounded_api_rate"] = _grounded_api_rate(
                                conn, answer.symbols_used
                            )
                            record["n_citations"] = len(answer.citations)
                        else:
                            answer = answer_question(q["question"])
                        record["answer"] = answer.model_dump()
                        record["checks"] = run_checks(answer)
                    except Exception as exc:  # noqa: BLE001 — record & continue, never lose the run
                        print(f"  failed: {type(exc).__name__}: {exc}")
                        record["error"] = str(exc)
                records.append(record)
                if "checks" in record:
                    rows.append((record["id"], record["checks"]))
                _flush(records, RESULTS)  # persist after every question
    finally:
        if conn is not None:
            conn.close()

    if rows:
        print("\n" + format_table(rows))
        passed = sum(1 for _, r in rows if all(v is None for v in r.values()))
        print(f"\n{passed}/{len(rows)} answers passed all checks → {RESULTS}")
        if GROUNDED:
            rates = [
                r["grounded_api_rate"]
                for r in records
                if r.get("grounded_api_rate") is not None
            ]
            cites = [r.get("n_citations", 0) for r in records if "answer" in r]
            if rates:
                avg = sum(rates) / len(rates)
                print(f"grounded_api_rate: {avg:.0%} (avg over {len(rates)} answers)")
            if cites:
                print(f"avg citations per answer: {sum(cites) / len(cites):.1f}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
