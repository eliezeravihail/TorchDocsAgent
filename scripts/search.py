"""Query the live index from the command line — the retrieval acceptance test.

    python scripts/search.py "scaled_dot_product_attention"
    python scripts/search.py "how to resize images" --library vision
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--library", default=None)
    args = parser.parse_args()

    load_dotenv()
    from index.hydrate import hydrate_section
    from index.retrieve import retrieve

    results = retrieve(args.query, k=args.k, library=args.library)
    if not results:
        print("no results — is the index built?")
        return 1

    for rank, pointer in enumerate(results, start=1):
        anchor = f"#{pointer['anchor']}" if pointer["anchor"] else ""
        print(f"{rank}. [{pointer['library']}] {pointer['heading_path']}")
        print(f"   {pointer['url']}{anchor}")

    top = hydrate_section(results[0])
    if top:
        snippet = " ".join(top["content"].split())[:300]
        print(f"\n-- top hit content --\n{snippet}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
