"""Embed corpus chunks locally (bge-small on CPU) and upsert into Neon.

Local by decision: Gemini's free embedding quota (~100 items/day) would take
weeks for a 7K-chunk corpus. A small open model has no quota, no key, and no
cost — the whole corpus embeds in minutes on a CI runner's CPU, and the same
model embeds queries at answer time.

Overnight/CI-safe by construction:
- resumable: a chunk whose (chunk_key, content_hash) is already in the DB is
  skipped, and every batch commits — kill it anytime, re-run continues;
- content is embedded and indexed (tsvector) but never stored.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from ingest.chunk_docs import chunk_page
from ingest.crawl import CORPUS_DIR, load_page

EMBED_MODEL = os.environ.get("TORCHDOCS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge convention: queries get an instruction prefix, documents do not
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 128
MAX_EMBED_CHARS = 2000  # bge-small context is 512 tokens; beyond it is truncated anyway


def chunk_key(unit: dict) -> str:
    raw = f"{unit['url']}#{unit['anchor']}#{' > '.join(unit['heading_path'])}"
    return hashlib.sha256(raw.encode()).hexdigest()


def iter_corpus_units(corpus_dir: Path = CORPUS_DIR) -> Iterator[dict]:
    """Walk the snapshot and yield every chunk unit of every page."""
    for path in sorted(corpus_dir.rglob("*.md")):
        meta, body = load_page(path)
        yield from chunk_page(meta, body)


def batches(items: list, size: int = BATCH_SIZE) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@cache
def _model():
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {EMBED_MODEL} (first run downloads ~130MB)")
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks locally; unit-normalized vectors."""
    vectors = _model().encode(
        [t[:MAX_EMBED_CHARS] for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a search query (bge instruction prefix) for retrieve()."""
    return embed_texts([QUERY_PREFIX + text])[0]


def existing_hashes(conn) -> dict[str, str]:
    rows = conn.execute("select chunk_key, content_hash from chunks").fetchall()
    return dict(rows)


UPSERT = """
insert into chunks (chunk_key, url, anchor, page_title, heading_path, library,
                    kind, source_link, content_hash, index_version, embedding, tsv)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s))
on conflict (chunk_key) do update set
    page_title = excluded.page_title, heading_path = excluded.heading_path,
    library = excluded.library, kind = excluded.kind,
    source_link = excluded.source_link, content_hash = excluded.content_hash,
    index_version = excluded.index_version, embedding = excluded.embedding,
    tsv = excluded.tsv
"""


def build_index(index_version: str, corpus_dir: Path = CORPUS_DIR, embed_fn=None) -> dict:
    """Embed every new/changed chunk in the snapshot into Neon."""
    from index.db import connect, ensure_schema

    embed_fn = embed_fn or embed_texts
    units = list(iter_corpus_units(corpus_dir))
    with connect() as conn:
        ensure_schema(conn)
        known = existing_hashes(conn)
        todo = [u for u in units if known.get(chunk_key(u)) != u["content_hash"]]
        print(f"[embed] {len(units)} chunks in snapshot, {len(todo)} new/changed to embed")

        done = 0
        for batch in batches(todo):
            vectors = embed_fn([u["content"] for u in batch])
            with conn.cursor() as cur:
                for unit, vector in zip(batch, vectors, strict=True):
                    cur.execute(
                        UPSERT,
                        (
                            chunk_key(unit),
                            unit["url"],
                            unit["anchor"],
                            unit["page_title"],
                            " > ".join(unit["heading_path"]),
                            unit["library"],
                            unit["kind"],
                            unit["source_link"],
                            unit["content_hash"],
                            index_version,
                            str(vector),
                            unit["content"],
                        ),
                    )
            conn.commit()  # checkpoint: safe to kill and re-run from here
            done += len(batch)
            print(f"[embed] {done}/{len(todo)} embedded")

        total = conn.execute("select count(*) from chunks").fetchone()[0]
    return {"snapshot_chunks": len(units), "embedded": len(todo), "db_total": total}
