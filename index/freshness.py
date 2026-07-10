"""Post-answer freshness: revalidate cited pages and self-heal the index.

Stale-while-revalidate for the docs index (the standard pattern: serve the
stored copy instantly, revalidate right after). Answers are served straight
from the chunks table (index/hydrate.py); AFTER an answer goes out, the pages
it cited are re-fetched live and compared chunk-by-chunk against the stored
content. Chunks that drifted get their `content` updated in place — so the
index self-heals exactly where users are looking — and the caller is told
which cited urls drifted so it can regenerate the just-shown answer from the
fresh text.

Deliberately NOT touched on drift: content_hash, embedding, tsv. The stored
hash must keep describing the text the embedding was computed from — the
weekly Build Index crawl then sees live-hash ≠ stored-hash and re-embeds the
page properly. Updating the hash here would make that crawl SKIP the row and
leave a stale embedding forever. Likewise, sections that appeared/vanished on
the live page (restructured headings) are left to the crawl: they have no
embedding to serve under, so an in-place content update cannot represent them.

A per-process TTL keeps one hot page from being re-fetched on every question,
and everything fails open — a freshness error can never break an already-shown
answer. Kill switch: TORCHDOCS_FRESHNESS=0.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext


# how long a checked URL stays "fresh enough" before the next question on it
# triggers another live comparison. Docs move slowly (weekly crawl is the
# backstop); an hour bounds both the fetch traffic and the staleness window.
def _ttl() -> float:
    return float(os.environ.get("TORCHDOCS_FRESHNESS_TTL_SECONDS", "3600"))


def enabled() -> bool:
    return os.environ.get("TORCHDOCS_FRESHNESS", "1") != "0"


_LOCK = threading.Lock()
_checked: dict[str, float] = {}  # url → monotonic time of the last live check


def _due(url: str) -> bool:
    """True once per TTL window per url (check-and-set, thread-safe)."""
    now = time.monotonic()
    with _LOCK:
        last = _checked.get(url)
        if last is not None and now - last < _ttl():
            return False
        _checked[url] = now
        if len(_checked) > 4096:  # bound the table across a long-lived process
            for key in [k for k, t in _checked.items() if now - t >= _ttl()]:
                del _checked[key]
        return True


def _live_units(url: str) -> list[dict]:
    """The page as it looks RIGHT NOW, chunked exactly like the index build."""
    from ingest.chunk_docs import chunk_page
    from ingest.crawl import extract_main_html, to_markdown
    from ingest.discover import fetch_html

    html = fetch_html(url)
    title, main = extract_main_html(html)
    return chunk_page({"url": url, "title": title}, to_markdown(main))


def refresh_pages(urls: list[str], conn=None) -> set[str]:
    """Revalidate these pages against the live docs; return the drifted urls.

    For each url not checked within the TTL: fetch the live page, chunk it the
    same way the build does, and compare each chunk's content to the stored
    row (matched by chunk_key). Drifted rows get `content` updated in place —
    content_hash/embedding/tsv stay as they are (see module docstring). Every
    failure is logged and skipped: freshness is best-effort by contract.
    """
    changed: set[str] = set()
    due = [u for u in dict.fromkeys(urls) if u and _due(u)]
    if not due:
        return changed

    from index.db import get_pool
    from index.embed import chunk_key

    ctx = nullcontext(conn) if conn is not None else get_pool().connection()
    try:
        with ctx as conn:
            for url in due:
                try:
                    live = _live_units(url)
                except Exception as exc:  # noqa: BLE001 — a dead page is not our problem here
                    print(f"[freshness] live fetch failed for {url}: {exc}", flush=True)
                    continue
                rows = dict(
                    conn.execute(
                        "select chunk_key, content from chunks where url = %s", (url,)
                    ).fetchall()
                )
                updates = [
                    (unit["content"], key)
                    for unit in live
                    if (key := chunk_key(unit)) in rows and rows[key] != unit["content"]
                ]
                if not updates:
                    continue
                with conn.cursor() as cur:
                    cur.executemany(
                        "update chunks set content = %s where chunk_key = %s", updates
                    )
                conn.commit()
                changed.add(url)
                print(
                    f"[freshness] {url}: {len(updates)} chunk(s) drifted; content refreshed",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 — never let freshness take an answer down
        print(f"[freshness] pass failed ({type(exc).__name__}: {exc}); skipping", flush=True)
    return changed


def reset() -> None:
    """Forget the TTL table (tests)."""
    with _LOCK:
        _checked.clear()
