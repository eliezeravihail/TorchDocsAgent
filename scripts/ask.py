"""Ask the agent one question, end to end (runs the agent tool loop).

    python scripts/ask.py "how do I build a CNN to classify images?"
    python scripts/ask.py "how is conv2d implemented?"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()

    load_dotenv()
    from agent.loop import answer_agentic

    answer = answer_agentic(args.question)

    print("\n" + "=" * 70)
    print(answer.answer_md)
    if answer.citations:
        print("\nCitations:")
        for c in answer.citations:
            anchor = f"#{c.anchor}" if c.anchor else ""
            print(f"  - {c.title or c.url}{anchor}\n    {c.url}{anchor}")
    if answer.referrals:
        print("\nBeyond these docs, see:")
        for r in answer.referrals:
            print(f"  - {r.url}  ({r.reason})")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
