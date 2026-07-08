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
import threading
from collections.abc import Iterator
from pathlib import Path

from ingest.chunk_docs import chunk_page
from ingest.crawl import CORPUS_DIR, load_page

EMBED_MODEL = os.environ.get("TORCHDOCS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge convention: queries get an instruction prefix, documents do not
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 128
MAX_EMBED_CHARS = 2000  # bge context is 512 tokens; beyond it is truncated anyway

# bump when indexed_text() changes → forces a one-time full re-embed (dims same,
# so the row-skip check would otherwise keep stale vectors). Reverting bge-base
# → bge-small changes dims too (768→384), which rebuilds the table outright —
# the stamp just keeps index_meta honest about which recipe is live.
EMBED_RECIPE = "v4-bge-small"


def chunk_key(unit: dict) -> str:
    raw = f"{unit['url']}#{unit['anchor']}#{' > '.join(unit['heading_path'])}"
    # part 0 keeps the legacy key format on purpose: existing rows stay valid
    # and the (chunk_key, content_hash) skip still holds, so introducing
    # size-capped parts embeds ONLY the new part-rows — no full re-embed.
    if unit.get("part", 0):
        raw += f"#part{unit['part']}"
    return hashlib.sha256(raw.encode()).hexdigest()


def symbol_from_url(url: str) -> str:
    """The qualified symbol an API page documents, from its filename.

    generated/torch.nn.functional.scaled_dot_product_attention.html
      → torch.nn.functional.scaled_dot_product_attention
    """
    stem = url.rsplit("/", 1)[-1].removesuffix(".html")
    root = stem.split(".")[0]
    return stem if ("." in stem and root in {"torch", "torchvision", "torchaudio"}) else ""


def indexed_text(unit: dict) -> str:
    """What we embed AND tsvector: symbol + heading path prepended to the body.

    Gives the URL/title strong semantic + lexical weight, so a short API page
    represents its symbol well instead of losing to verbose tutorials.
    """
    parts = []
    symbol = symbol_from_url(unit["url"])
    if symbol:
        parts.append(symbol)
    heading = " > ".join(unit.get("heading_path", []))
    if heading:
        parts.append(heading)
    parts.append(unit["content"])
    return "\n".join(parts)


def iter_corpus_units(corpus_dir: Path = CORPUS_DIR) -> Iterator[dict]:
    """Walk the snapshot and yield every chunk unit of every page."""
    for path in sorted(corpus_dir.rglob("*.md")):
        meta, body = load_page(path)
        yield from chunk_page(meta, body)


def batches(items: list, size: int = BATCH_SIZE) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


_MODEL_LOCK = threading.Lock()
_MODEL = None


def _model():
    # Double-checked load: functools.cache would let two concurrent first
    # queries (before _warm_up finishes, or when it is skipped) both enter the
    # body and load the 130MB model twice. Guard the build with a lock and a
    # re-check so it happens exactly once; encode() itself is safe to call
    # concurrently on the shared instance.
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                print(f"[embed] loading {EMBED_MODEL} (first run downloads ~130MB)")
                _MODEL = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _MODEL


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
                    kind, source_link, content_hash, index_version, part,
                    embedding, tsv)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s))
on conflict (chunk_key) do update set
    page_title = excluded.page_title, heading_path = excluded.heading_path,
    library = excluded.library, kind = excluded.kind,
    source_link = excluded.source_link, content_hash = excluded.content_hash,
    index_version = excluded.index_version, part = excluded.part,
    embedding = excluded.embedding, tsv = excluded.tsv
"""


def build_index(index_version: str, corpus_dir: Path = CORPUS_DIR, embed_fn=None) -> dict:
    """Embed every new/changed chunk in the snapshot into Neon."""
    from index.db import connect, ensure_schema

    embed_fn = embed_fn or embed_texts
    units = list(iter_corpus_units(corpus_dir))
    with connect() as conn:
        ensure_schema(conn)
        from index.db import get_meta, set_meta

        known = existing_hashes(conn)
        db_keys = set(known)  # real rows in the DB, used for the stale purge below
        if get_meta(conn, "embed_recipe") != EMBED_RECIPE:
            print(f"[embed] embed recipe changed → full re-embed ({EMBED_RECIPE})")
            known = {}  # ignore existing hashes so every chunk is re-embedded once
        todo = [u for u in units if known.get(chunk_key(u)) != u["content_hash"]]
        print(f"[embed] {len(units)} chunks in snapshot, {len(todo)} new/changed to embed")

        done = 0
        for batch in batches(todo):
            vectors = embed_fn([indexed_text(u) for u in batch])
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
                            unit.get("part", 0),
                            str(vector),
                            indexed_text(unit),
                        ),
                    )
            conn.commit()  # checkpoint: safe to kill and re-run from here
            done += len(batch)
            print(f"[embed] {done}/{len(todo)} embedded")

        # purge rows whose chunk no longer exists in the snapshot (renamed
        # headings, deleted pages) — dead pointers must not win retrieval.
        # Purge off db_keys, not `known`: a recipe bump zeroes `known`, and
        # purging off it would silently never delete anything.
        live_keys = {chunk_key(u) for u in units}
        stale = [k for k in db_keys if k not in live_keys]
        for batch in batches(stale, 500):
            conn.execute("delete from chunks where chunk_key = any(%s)", (batch,))
        conn.commit()
        if stale:
            print(f"[embed] purged {len(stale)} stale chunks")

        # stamp the recipe only after a full pass, so a mid-run death re-forces
        # the full re-embed next time instead of leaving mixed vector recipes
        set_meta(conn, "embed_recipe", EMBED_RECIPE)
        conn.commit()

        total = conn.execute("select count(*) from chunks").fetchone()[0]
    return {"snapshot_chunks": len(units), "embedded": len(todo), "db_total": total}
