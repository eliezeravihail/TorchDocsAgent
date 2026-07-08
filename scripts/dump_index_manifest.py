"""Dump a verified manifest of everything in the live index.

Usage:  python scripts/dump_index_manifest.py   (needs NEON_URL)

Writes eval/index_manifest.jsonl — one line per distinct doc PAGE actually in
the chunks table, with the fields needed to author a grounded eval set whose
"expected sources" are, by construction, retrievable:

    {"url", "kind", "library", "page_title", "n_chunks", "headings": [...]}

This is the ground truth the eval questions are built from — no page, symbol,
or URL is invented from memory; every question maps to a page that exists here.
Run it from Actions (the dev sandbox can't reach Neon) and commit the result.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

OUT = Path(__file__).parent.parent / "eval" / "index_manifest.jsonl"

# per-page rollup: title/kind/library are constant per url; headings vary by row
PAGE_SQL = """
select url, kind, library, page_title, heading_path
from chunks
order by library, kind, url
"""


def main() -> int:
    load_dotenv()
    from index.db import connect

    pages: dict[str, dict] = {}
    headings: dict[str, list[str]] = defaultdict(list)
    with connect() as conn:
        for url, kind, library, title, heading in conn.execute(PAGE_SQL).fetchall():
            pages.setdefault(
                url, {"url": url, "kind": kind, "library": library, "page_title": title}
            )
            if heading:
                headings[url].append(heading)

    OUT.parent.mkdir(exist_ok=True)
    by_kind: dict[str, int] = defaultdict(int)
    with OUT.open("w") as out:
        for url, meta in pages.items():
            hs = headings[url]
            record = {**meta, "n_chunks": len(hs) or 1, "headings": sorted(set(hs))[:40]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            by_kind[meta["kind"]] += 1

    print(f"{len(pages)} distinct pages → {OUT}")
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind or '(none)':<10} {n}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
