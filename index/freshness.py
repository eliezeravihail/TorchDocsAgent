"""Post-answer freshness: revalidate cited pages and self-heal the index.

Stale-while-revalidate for the docs index (the standard pattern: serve the
stored copy instantly, revalidate right after). Answers are served straight
from the chunks table (index/hydrate.py); AFTER an answer goes out, the pages
it cited are re-fetched live and compared chunk-by-chunk against the stored
content. A drifted chunk is fixed COMPLETELY in place: content, content_hash,
embedding, and tsvector — the embedding model is already hot in this process
(it embeds every incoming query), so re-embedding a handful of chunks costs
milliseconds and leaves the row fully consistent immediately. The caller is
told which cited urls drifted so it can regenerate the just-shown answer.

The re-embed reuses the page's EXISTING gloss and hypothetical questions
(indexed_text folds them in from the committed files): they describe what the
symbol is for, which small doc edits don't change. Structural changes — a new
page, a deleted page, restructured sections, or drift drastic enough to need
fresh enrichment — stay the job of the periodic Build Index crawl; sections
that appeared/vanished on the live page are therefore skipped here. Because
the stored hash now matches the re-embedded text, that crawl correctly skips
the healed rows instead of re-embedding them again.

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
    """The page as it looks RIGHT NOW, chunked exactly like the index build.

    The page-level content_hash is computed the same way the crawl does
    (sha256 of the markdown body — ingest/crawl.save_page), so the healed rows
    carry the hash the next crawl will compute and be correctly skipped by it.
    """
    import hashlib

    from ingest.chunk_docs import chunk_page
    from ingest.crawl import extract_main_html, to_markdown
    from ingest.discover import fetch_html

    html = fetch_html(url)
    title, main = extract_main_html(html)
    body = to_markdown(main)
    meta = {
        "url": url,
        "title": title,
        "content_hash": hashlib.sha256(body.encode()).hexdigest(),
    }
    return chunk_page(meta, body)


# the full heal: the row leaves this statement exactly as a fresh build would
# have written it — text, hash, vector, and keyword index all describe the
# same (new) content, so retrieval and answers agree from the next query on
_HEAL = """
update chunks
set content = %s, content_hash = %s, embedding = %s,
    tsv = to_tsvector('english', %s)
where chunk_key = %s
"""


def refresh_pages(urls: list[str], conn=None) -> set[str]:
    """Revalidate these pages against the live docs; return the drifted urls.

    For each url not checked within the TTL: fetch the live page, chunk it the
    same way the build does, and compare each chunk's content to the stored
    row (matched by chunk_key). Drifted chunks are healed in place — content,
    content_hash, a freshly computed embedding (the model is already loaded in
    this process), and the tsvector, with indexed_text reusing the page's
    existing gloss/questions. Every failure is logged and skipped: freshness
    is best-effort by contract.
    """
    changed: set[str] = set()
    due = [u for u in dict.fromkeys(urls) if u and _due(u)]
    if not due:
        return changed

    from index.db import get_pool
    from index.embed import chunk_key, embed_texts, indexed_text

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
                drifted = [
                    (key, unit)
                    for unit in live
                    if (key := chunk_key(unit)) in rows and rows[key] != unit["content"]
                ]
                if not drifted:
                    continue
                texts = [indexed_text(unit) for _, unit in drifted]
                vectors = embed_texts(texts)
                with conn.cursor() as cur:
                    for (key, unit), text, vector in zip(drifted, texts, vectors, strict=True):
                        cur.execute(
                            _HEAL,
                            (unit["content"], unit["content_hash"], str(vector), text, key),
                        )
                conn.commit()
                changed.add(url)
                print(
                    f"[freshness] {url}: {len(drifted)} chunk(s) drifted; "
                    "content + embedding healed in place",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 — never let freshness take an answer down
        print(f"[freshness] pass failed ({type(exc).__name__}: {exc}); skipping", flush=True)
    return changed


def reset() -> None:
    """Forget the TTL table (tests)."""
    with _LOCK:
        _checked.clear()
