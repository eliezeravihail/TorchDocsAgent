"""Read actual content for a pointer — from the crawl snapshot, or live.

Two granularities, matching the agent's tools:
- hydrate_section: one retrieved chunk's text (for search_docs results)
- hydrate_page: a whole page, outline-first if oversized (for read_page)

When the snapshot file is absent (e.g. a deployed app with no bundled
_corpus/), fall back to fetching the live page. This keeps the app
self-contained without shipping the snapshot; set TORCHDOCS_LIVE_HYDRATE=0
to disable and require the snapshot.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ingest.chunk_docs import chunk_page, split_by_heading
from ingest.crawl import CORPUS_DIR, load_page, page_path

PAGE_CHAR_LIMIT = 30_000  # beyond this, read_page returns the outline first
_LIVE = os.environ.get("TORCHDOCS_LIVE_HYDRATE", "1") != "0"


@lru_cache(maxsize=256)
def _live_page(url: str) -> tuple[dict, str] | None:
    """Fetch, strip, and convert a live doc page → (meta, markdown body). Cached."""
    if not _LIVE:
        return None
    from ingest.crawl import extract_main_html, to_markdown
    from ingest.discover import fetch_html

    try:
        # follow client-side redirects (docs/stable/... → versioned page); a
        # plain fetch would return the "Redirecting…" stub and hollow the answer
        html = fetch_html(url)
    except Exception as exc:  # noqa: BLE001 — a dead link just means "no content"
        print(f"[hydrate] live fetch failed for {url}: {exc}", flush=True)
        return None
    title, main = extract_main_html(html)
    body = to_markdown(main)
    return {"url": url, "title": title, "content": body}, body


def _load(url: str, corpus_dir: Path) -> tuple[dict, str] | None:
    """Snapshot first; live fetch as fallback."""
    path = page_path(url, corpus_dir)
    if path.exists():
        return load_page(path)
    return _live_page(url)


def hydrate_section(pointer: dict, corpus_dir: Path = CORPUS_DIR) -> dict | None:
    """Return the pointer enriched with its section content, or None if gone.

    If the pointer's heading no longer matches any section (the page changed,
    or the heading was reformatted), we return None — the section is treated as
    gone. We deliberately do NOT substitute the page preamble: that would show
    unrelated text under the section's citation, which breaks the grounding
    contract silently.
    """
    # fast path: retrieve() now returns each chunk's stored `content` in the
    # pointer, so the section is already in hand — no snapshot read, no live
    # fetch (the per-section fetch was the dominant answer latency). Empty
    # content (a row not yet backfilled) falls through to the fetch path below,
    # so this is safe during the migration.
    if pointer.get("content"):
        return dict(pointer)

    loaded = _load(pointer["url"], corpus_dir)
    if loaded is None:
        return None
    meta, body = loaded
    heading_path = pointer.get("heading_path", "")
    part = pointer.get("part") or 0  # size-split sections: which part of the section
    for unit in chunk_page(meta, body):
        if " > ".join(unit["heading_path"]) == heading_path and unit.get("part", 0) == part:
            return {**pointer, "content": unit["content"]}
    print(
        f"[hydrate] heading {heading_path!r} (part {part}) not found in {pointer['url']} "
        "(page changed); dropping the pointer",
        flush=True,
    )
    return None


def hydrate_sections(
    pointers: list[dict], corpus_dir: Path = CORPUS_DIR, max_workers: int = 8
) -> list[dict]:
    """Hydrate many pointers CONCURRENTLY, preserving retrieval order.

    On a deployed Space (no bundled snapshot) each hydrate_section does a live
    page fetch; doing k of them one-after-another was the dominant answer
    latency (measured p50≈12s, one outlier 69s). The fetches are I/O-bound and
    independent, so a thread pool collapses k network round-trips into roughly
    one. Retrieval order is preserved (rank feeds the answer context) and gone
    sections (None) are dropped, so the result is identical to the sequential
    comprehension it replaces — only faster.
    """
    from concurrent.futures import ThreadPoolExecutor

    pointers = list(pointers)
    if not pointers:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(pointers))) as pool:
        hydrated = pool.map(lambda p: hydrate_section(p, corpus_dir), pointers)
    return [s for s in hydrated if s]


def hydrate_page(url: str, corpus_dir: Path = CORPUS_DIR) -> dict | None:
    """Whole page markdown; oversized pages return their heading outline instead."""
    loaded = _load(url, corpus_dir)
    if loaded is None:
        return None
    meta, body = loaded
    if len(body) > PAGE_CHAR_LIMIT:
        outline = [
            " > ".join(section.heading_path)
            for section in split_by_heading(body)
            if section.title
        ]
        return {"url": url, "title": meta.get("title", ""), "outline": outline}
    return {"url": url, "title": meta.get("title", ""), "content": body}
