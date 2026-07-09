"""Fetch rendered doc pages, strip chrome, convert to markdown, snapshot to disk.

The on-disk snapshot (``_corpus/``) is the source of truth for the index:
one .md file per page with YAML frontmatter (url, title, library,
content_hash, crawled_at). Unchanged content_hash ⇒ the page is skipped by
re-chunking/re-embedding — that is what makes the weekly recrawl cheap.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

CORPUS_DIR = Path("_corpus")

# Sphinx-theme content containers, most specific first; <body> is the fallback.
_MAIN_SELECTORS = ["article.pytorch-article", "div[role=main]", "main", "article", "body"]
_CHROME_TAGS = ["nav", "header", "footer", "script", "style", "aside", "form", "iframe"]


def extract_main_html(html: str) -> tuple[str, str]:
    """Return (title, main-content html) with navigation chrome removed."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for selector in _MAIN_SELECTORS:
        main = soup.select_one(selector)
        if main is not None:
            break
    else:  # pragma: no cover — soup always has some root
        main = soup
    for tag in main.find_all(_CHROME_TAGS):
        tag.decompose()
    return title, str(main)


def to_markdown(html_fragment: str) -> str:
    from markdownify import markdownify

    return markdownify(html_fragment, heading_style="ATX", bullets="*").strip()


def page_path(url: str, corpus_dir: Path = CORPUS_DIR) -> Path:
    """Stable on-disk location for a page: host dropped, path preserved."""
    parsed = urlparse(url)
    relative = parsed.path.strip("/") or "index"
    if relative.endswith(".html"):
        relative = relative[: -len(".html")]
    return corpus_dir / f"{relative}.md"


def load_page(path: Path) -> tuple[dict, str]:
    """Read one snapshot file → (frontmatter dict, markdown body)."""
    _, frontmatter, body = path.read_text(encoding="utf-8").split("---\n", 2)
    return yaml.safe_load(frontmatter), body.lstrip("\n")


def save_page(
    url: str,
    library: str,
    html: str,
    corpus_dir: Path = CORPUS_DIR,
) -> bool:
    """Convert and write one page; returns True if content changed on disk."""
    title, main_html = extract_main_html(html)
    body = to_markdown(main_html)
    content_hash = hashlib.sha256(body.encode()).hexdigest()

    path = page_path(url, corpus_dir)
    if path.exists() and load_page(path)[0].get("content_hash") == content_hash:
        return False  # unchanged — nothing downstream needs to re-run

    meta = {
        "url": url,
        "title": title,
        "library": library,
        "content_hash": content_hash,
        "crawled_at": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\n{yaml.safe_dump(meta, sort_keys=True)}---\n\n{body}\n", encoding="utf-8"
    )
    return True


def crawl(
    pages: dict[str, set[str]],
    corpus_dir: Path = CORPUS_DIR,
) -> dict[str, int]:
    """Fetch and snapshot every discovered page; returns per-library change counts.

    Politeness: a fixed delay between requests (TORCHDOCS_CRAWL_DELAY seconds,
    default 0.2) keeps a thousands-page crawl from hammering docs.pytorch.org —
    both to be a good citizen and to not get the CI runner's IP blocked.
    """
    import os
    import time

    from ingest.discover import fetch_html

    delay = float(os.environ.get("TORCHDOCS_CRAWL_DELAY", "0.2"))
    changed: dict[str, int] = {}
    for library, urls in pages.items():
        count = 0
        for url in sorted(urls):
            try:
                # follow client-side redirects: docs/stable/... serves a
                # "Redirecting…" stub whose real content is behind a meta-refresh
                html = fetch_html(url)
            except Exception as exc:  # noqa: BLE001 — one bad page must not kill the crawl
                print(f"[crawl] {url}: fetch failed ({exc})")
                continue
            finally:
                if delay > 0:
                    time.sleep(delay)
            if save_page(url, library, html, corpus_dir):
                count += 1
        changed[library] = count
        print(f"[crawl] {library}: {count} pages changed")
    return changed
