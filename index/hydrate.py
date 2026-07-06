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
    from ingest.discover import fetch

    try:
        html = fetch(url).decode("utf-8", errors="replace")
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
    live = _live_page(url)
    return (live[0], live[1]) if live else None


def hydrate_section(pointer: dict, corpus_dir: Path = CORPUS_DIR) -> dict | None:
    """Return the pointer enriched with its section content, or None if gone."""
    loaded = _load(pointer["url"], corpus_dir)
    if loaded is None:
        return None
    meta, body = loaded
    heading_path = pointer.get("heading_path", "")
    for unit in chunk_page(meta, body):
        if " > ".join(unit["heading_path"]) == heading_path:
            return {**pointer, "content": unit["content"]}
    # heading not found live (page changed): fall back to the page preamble
    first = next(iter(chunk_page(meta, body)), None)
    return {**pointer, "content": first["content"]} if first else None


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
