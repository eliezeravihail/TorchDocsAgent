"""End-to-end index build: discover → crawl → chunk+embed. Overnight-safe.

Embeddings run locally on CPU (no API key needed); only NEON_URL is required.
Usage (from the repo root, with .env holding NEON_URL):

    python scripts/build_index.py                 # full pipeline
    python scripts/build_index.py --skip-crawl    # re-embed existing snapshot only
    python scripts/build_index.py --libraries core,tutorials

Every stage is resumable: crawling skips pages whose content is unchanged,
embedding skips chunks already in the DB with the same hash, and every
embed batch commits. Kill it anytime; re-running continues where it stopped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-crawl", action="store_true", help="embed the existing snapshot")
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="crawl only, refresh the snapshot without embedding (e.g. to gloss before embed)",
    )
    parser.add_argument("--libraries", default="", help="comma-separated subset of the seed list")
    args = parser.parse_args()

    load_dotenv()
    # fail fast: better to stop here than crash after an hour of crawling.
    # crawl-only (--skip-embed) never touches Neon, so don't demand the URL for it.
    if not args.skip_embed and not os.environ.get("NEON_URL"):
        print("ERROR: missing required env var NEON_URL (see .env.example)")
        return 1
    from index.embed import build_index
    from ingest.crawl import crawl
    from ingest.discover import SEEDS, discover

    started = time.time()
    index_version = datetime.now(UTC).strftime("crawl-%Y%m%d-%H%M")

    if not args.skip_crawl:
        seeds = SEEDS
        if args.libraries:
            wanted = {name.strip() for name in args.libraries.split(",")}
            seeds = {k: v for k, v in SEEDS.items() if k in wanted}
        print(f"== discover: {', '.join(seeds)}")
        pages = discover(seeds)
        print(f"== crawl: {sum(len(v) for v in pages.values())} pages")
        crawl(pages)

    if args.skip_embed:
        minutes = (time.time() - started) / 60
        print(f"\n== CRAWL-ONLY DONE in {minutes:.0f} min: snapshot refreshed, no embed")
        return 0

    print(f"== embed → Neon (index_version={index_version})")
    stats = build_index(index_version)

    minutes = (time.time() - started) / 60
    print(
        f"\n== DONE in {minutes:.0f} min: {stats['snapshot_chunks']} chunks in snapshot, "
        f"{stats['embedded']} embedded this run, {stats['db_total']} total in DB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
