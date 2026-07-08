"""Neon connection and the chunks schema.

Pointer-based by design: no page text column. Content lives in the
_corpus/ snapshot; the DB holds vectors, tsvectors, and pointers.

Two ways to reach Neon, for two workloads:
- `connect()` — one dedicated connection, for the batch index build (a single
  long-running writer that commits in checkpoints).
- `get_pool()` — a shared connection pool, for the web app, where many
  concurrent questions each borrow a connection for a couple of quick reads.
  Reconnecting per read (TLS handshake ~100-300ms) would dominate answer time
  under load and risk exhausting Neon's free-tier connection cap.
"""

from __future__ import annotations

import os
from functools import cache

import psycopg

# Embedding width, derived from the model so it can't drift from index/embed.py.
# bge-base-en-v1.5 (768d) resolves finer semantic distinctions than bge-small
# (384d) — a "loss function" question lands nearer the reference page instead
# of tutorial prose — at ~4x the model size, still local/CPU/free. A model not
# in the table needs TORCHDOCS_EMBED_DIMS set. Changing dims rebuilds the
# chunks table automatically (see ensure_schema).
_MODEL_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}
_DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def embed_dims() -> int:
    """The embedding width for the configured model.

    Reads the same env var as index/embed.py's EMBED_MODEL, so the schema
    width and the vectors written into it can never disagree. An override
    (TORCHDOCS_EMBED_DIMS) covers a model not in the table; an unknown model
    with no override is a loud config error, not a silently wrong-width table.
    """
    override = os.environ.get("TORCHDOCS_EMBED_DIMS")
    if override:
        return int(override)
    model = os.environ.get("TORCHDOCS_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
    if model not in _MODEL_DIMS:
        raise RuntimeError(
            f"unknown embed model {model!r}; set TORCHDOCS_EMBED_DIMS to its width"
        )
    return _MODEL_DIMS[model]


EMBED_DIMS = embed_dims()

SCHEMA = f"""
create extension if not exists vector;

create table if not exists chunks (
    id            bigserial primary key,
    chunk_key     text not null unique,   -- sha256(url#anchor#heading_path)
    url           text not null,
    anchor        text not null default '',
    page_title    text not null default '',
    heading_path  text not null default '',
    library       text not null default '',
    kind          text not null default '',
    source_link   text not null default '',
    content_hash  text not null,
    index_version text not null,
    part          int not null default 0,  -- ordinal within a size-split section
    embedding     vector({EMBED_DIMS}) not null,
    tsv           tsvector not null
);

create index if not exists chunks_embedding_idx
    on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_tsv_idx
    on chunks using gin (tsv);
create index if not exists chunks_url_idx on chunks (url);

create table if not exists index_meta (
    key   text primary key,
    value text not null
);
"""


def _neon_url() -> str:
    url = os.environ.get("NEON_URL")
    if not url:
        raise RuntimeError("NEON_URL is not set (see .env.example)")
    return url


def connect() -> psycopg.Connection:
    return psycopg.connect(_neon_url())


@cache
def get_pool():
    """Process-wide pooled access to Neon, built lazily on first use.

    Cached so every request shares one pool. `max_size` caps concurrent DB
    connections (keep it at or under Neon's plan limit); it need not equal the
    app's request concurrency — a request holds a connection only for the brief
    reads inside retrieve(), then returns it. `check` validates a connection on
    checkout so a Neon-side idle timeout surfaces as a fresh connection, not a
    query error mid-request.
    """
    from psycopg_pool import ConnectionPool

    max_size = int(os.environ.get("TORCHDOCS_DB_POOL", "8"))
    pool = ConnectionPool(
        _neon_url(),
        min_size=1,
        max_size=max_size,
        check=ConnectionPool.check_connection,
        open=False,  # constructor-time open is deprecated in psycopg_pool 3.2
    )
    pool.open()
    # the app SELECTs columns that may postdate the live table (e.g. `part`);
    # apply the idempotent migrations here too, or a fresh deploy 500s on every
    # search until the next index build runs ensure_schema
    with pool.connection() as conn:
        for migration in RUNTIME_MIGRATIONS:
            conn.execute(migration)
    return pool


def get_meta(conn: psycopg.Connection, key: str) -> str | None:
    row = conn.execute("select value from index_meta where key = %s", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: psycopg.Connection, key: str, value: str) -> None:
    conn.execute(
        "insert into index_meta (key, value) values (%s, %s) "
        "on conflict (key) do update set value = excluded.value",
        (key, value),
    )


# Columns added after the chunks table first shipped. CREATE IF NOT EXISTS
# won't touch an existing table, so both writers (ensure_schema) and the app
# (get_pool) apply these idempotent migrations — the app because it may deploy
# and SELECT a new column before the next index build ever runs.
RUNTIME_MIGRATIONS = [
    "alter table chunks add column if not exists part int not null default 0",
]


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    for migration in RUNTIME_MIGRATIONS:
        conn.execute(migration)
    # the index is a rebuildable cache: if the embedding dimension changed
    # (model swap), drop and recreate rather than mixing vector spaces
    row = conn.execute(
        "select atttypmod from pg_attribute "
        "where attrelid = 'chunks'::regclass and attname = 'embedding'"
    ).fetchone()
    if row and row[0] != EMBED_DIMS:
        print(f"[db] embedding dims changed ({row[0]} -> {EMBED_DIMS}); rebuilding chunks table")
        conn.execute("drop table chunks")
        conn.execute(SCHEMA)
    conn.commit()
