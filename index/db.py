"""Neon connection and the chunks schema.

Pointer-based by design: no page text column. Content lives in the
_corpus/ snapshot; the DB holds vectors, tsvectors, and pointers.
"""

from __future__ import annotations

import os

import psycopg

EMBED_DIMS = 768  # gemini-embedding-001 truncated output; vectors re-normalized in code

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
"""


def connect() -> psycopg.Connection:
    url = os.environ.get("NEON_URL")
    if not url:
        raise RuntimeError("NEON_URL is not set (see .env.example)")
    return psycopg.connect(url)


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()
