"""Dump the AUTHORITATIVE PyTorch docs inventory — external ground truth.

Usage:  python scripts/dump_docs_inventory.py   (needs network to docs.pytorch.org)

Reads what the documentation SITE itself publishes, independent of our crawl,
chunking, or embedding: the Sphinx ``objects.inv`` per doc set (every
documented symbol → exact page URL + anchor) plus the tutorial sitemap. This
is the ground truth the eval questions are built from, so the eval can catch
pipeline flaws — a page that exists here but is missing from our index
(eval/index_manifest.jsonl) is a crawl/index bug, not a retrieval one.

Must run where docs.pytorch.org is reachable (Actions), NOT the dev sandbox
(egress policy blocks it). Writes eval/docs_inventory.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "eval" / "docs_inventory.jsonl"

# roles worth authoring questions against: the actual API surface + tutorial
# pages (std:doc). Skips std:label / genindex / py:module noise.
_KEEP_ROLES = {
    "py:class",
    "py:function",
    "py:method",
    "py:attribute",
    "py:data",
    "std:doc",
}


def main() -> int:
    import requests

    from ingest.discover import SEEDS, fetch, parse_objects_inv, parse_sitemap

    records: list[dict] = []
    for library, base in SEEDS.items():
        n_before = len(records)
        try:
            inv = fetch(base + "objects.inv")
            for e in parse_objects_inv(inv, base):
                if e.role in _KEEP_ROLES and e.page_url.endswith(".html"):
                    records.append(
                        {
                            "library": library,
                            "name": e.name,
                            "role": e.role,
                            "url": e.page_url,
                            "anchor": e.anchor,
                        }
                    )
        except requests.RequestException as exc:
            print(f"[inv] {library}: objects.inv unavailable ({exc})")
        try:
            sitemap = fetch(base + "sitemap.xml").decode("utf-8")
            for url in parse_sitemap(sitemap):
                if url.startswith(base) and url.endswith(".html"):
                    records.append(
                        {"library": library, "name": "", "role": "sitemap",
                         "url": url, "anchor": ""}
                    )
        except requests.RequestException as exc:
            print(f"[sitemap] {library}: sitemap unavailable ({exc})")
        print(f"[inv] {library}: {len(records) - n_before} entries")

    # de-dup (a symbol can appear in both inv and sitemap); keep the richer row
    seen: dict[tuple, dict] = {}
    for r in records:
        key = (r["url"], r["anchor"], r["name"])
        if key not in seen or r["role"] != "sitemap":
            seen[key] = r

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as out:
        for r in seen.values():
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(seen)} inventory entries → {OUT}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
