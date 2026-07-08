"""Coverage check: docs the SITE publishes vs. pages our index actually holds.

Usage:  python scripts/coverage_diff.py   (reads the two committed dumps)

Reports pages present in eval/docs_inventory.jsonl (external ground truth from
docs.pytorch.org) whose URL is absent from eval/index_manifest.jsonl (what our
crawl+index captured). A non-empty gap is a PIPELINE bug — a page the docs
site documents that our system can never retrieve — exactly what an eval built
only from the index would be blind to.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

EVAL = Path(__file__).parent.parent / "eval"
INVENTORY = EVAL / "docs_inventory.jsonl"
MANIFEST = EVAL / "index_manifest.jsonl"


def _urls(path: Path) -> set[str]:
    return {json.loads(line)["url"] for line in path.open() if line.strip()}


def main() -> int:
    if not INVENTORY.exists() or not MANIFEST.exists():
        print("need both eval/docs_inventory.jsonl and eval/index_manifest.jsonl")
        return 1
    site = [json.loads(line) for line in INVENTORY.open() if line.strip()]
    site_urls = {r["url"] for r in site}
    indexed = _urls(MANIFEST)

    missing = sorted(site_urls - indexed)
    by_lib = Counter(
        r["library"] for r in site if r["url"] in set(missing)
    )
    print(f"docs-site pages: {len(site_urls)}   indexed pages: {len(indexed)}")
    print(f"in the docs but NOT in our index: {len(missing)}")
    for lib, n in by_lib.most_common():
        print(f"  {lib:<12} {n}")
    for url in missing[:60]:
        print(f"    {url}")
    if len(missing) > 60:
        print(f"    … and {len(missing) - 60} more")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
