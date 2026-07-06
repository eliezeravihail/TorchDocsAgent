"""Neon connection and the chunks schema.

Pointer-based by design: no page text column. Content lives in the
_corpus/ snapshot; the DB holds vectors, tsvectors, and pointers.
"""

from __future__ import annotations

import os

import psycopg

EMBED_DIMS = 384  # BAAI/bge-small-en-v1.5 — local CPU model, no API quota

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


def connect() -> psycopg.Connection:
    url = os.environ.get("NEON_URL")
    if not url:
        raise RuntimeError("NEON_URL is not set (see .env.example)")
    return psycopg.connect(url)


def get_meta(conn: psycopg.Connection, key: str) -> str | None:
    row = conn.execute("select value from index_meta where key = %s", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: psycopg.Connection, key: str, value: str) -> None:
    conn.execute(
        "insert into index_meta (key, value) values (%s, %s) "
        "on conflict (key) do update set value = excluded.value",
        (key, value),
    )


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
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
