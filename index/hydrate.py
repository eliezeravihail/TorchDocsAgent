"""Read actual content from the crawl snapshot, given pointers.

Two granularities, matching the agent's tools:
- hydrate_section: one retrieved chunk's text (for search_docs results)
- hydrate_page: a whole page, outline-first if oversized (for read_page)
"""

from __future__ import annotations

from pathlib import Path

from ingest.chunk_docs import chunk_page, split_by_heading
from ingest.crawl import CORPUS_DIR, load_page, page_path

PAGE_CHAR_LIMIT = 30_000  # beyond this, read_page returns the outline first


def hydrate_section(pointer: dict, corpus_dir: Path = CORPUS_DIR) -> dict | None:
    """Return the pointer enriched with its section content, or None if gone."""
    path = page_path(pointer["url"], corpus_dir)
    if not path.exists():
        return None
    meta, body = load_page(path)
    heading_path = pointer.get("heading_path", "")
    for unit in chunk_page(meta, body):
        if " > ".join(unit["heading_path"]) == heading_path:
            return {**pointer, "content": unit["content"]}
    return None


def hydrate_page(url: str, corpus_dir: Path = CORPUS_DIR) -> dict | None:
    """Whole page markdown; oversized pages return their heading outline instead."""
    path = page_path(url, corpus_dir)
    if not path.exists():
        return None
    meta, body = load_page(path)
    if len(body) > PAGE_CHAR_LIMIT:
        outline = [
            " > ".join(section.heading_path)
            for section in split_by_heading(body)
            if section.title
        ]
        return {"url": url, "title": meta.get("title", ""), "outline": outline}
    return {"url": url, "title": meta.get("title", ""), "content": body}
