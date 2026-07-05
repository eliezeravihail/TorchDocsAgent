"""Embed corpus chunks with gemini-embedding-001 and upsert into Neon.

Overnight-safe by construction:
- resumable: a chunk whose (chunk_key, content_hash) is already in the DB is
  skipped, and every batch commits — kill it anytime, re-run continues;
- rate-limit aware: 429s wait out the free-tier window with long backoff;
- content is embedded and indexed (tsvector) but never stored.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Iterator
from pathlib import Path

from ingest.chunk_docs import chunk_page
from ingest.crawl import CORPUS_DIR, load_page

EMBED_MODEL = os.environ.get("TORCHDOCS_EMBED_MODEL", "gemini-embedding-001")
BATCH_SIZE = 64
MAX_EMBED_CHARS = 8000  # ~2k tokens, the embedding model's input ceiling


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


def normalize(vector: list[float]) -> list[float]:
    """Unit-normalize (required when truncating gemini embeddings' dims)."""
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed_texts(client, texts: list[str], retries: int = 8) -> list[list[float]]:
    """One batched embedding call with free-tier-window backoff."""
    from google.genai import errors, types

    config = types.EmbedContentConfig(output_dimensionality=768, task_type="RETRIEVAL_DOCUMENT")
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[t[:MAX_EMBED_CHARS] for t in texts],
                config=config,
            )
            return [normalize(e.values) for e in result.embeddings]
        except errors.APIError as exc:
            last_exc = exc
            wait = 30 * (attempt + 1) if exc.code == 429 else 2**attempt
            print(f"[embed] API error {exc.code}, waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
    raise RuntimeError(f"embedding failed after {retries} attempts: {last_exc}")


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


def build_index(index_version: str, corpus_dir: Path = CORPUS_DIR, client=None) -> dict:
    """Embed every new/changed chunk in the snapshot into Neon."""
    from google import genai

    from index.db import connect, ensure_schema

    if client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=key)

    units = list(iter_corpus_units(corpus_dir))
    with connect() as conn:
        ensure_schema(conn)
        known = existing_hashes(conn)
        todo = [u for u in units if known.get(chunk_key(u)) != u["content_hash"]]
        print(f"[embed] {len(units)} chunks in snapshot, {len(todo)} new/changed to embed")

        done = 0
        for batch in batches(todo):
            vectors = embed_texts(client, [u["content"] for u in batch])
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
