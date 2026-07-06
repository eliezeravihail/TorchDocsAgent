"""Hybrid retrieval over the chunks index: dense + keyword, RRF-merged.

Returns POINTERS (url, anchor, heading path, ...) — never content. The
caller hydrates content from the snapshot (index/hydrate.py). This is the
engine behind the agent's `search_docs` tool.
"""

from __future__ import annotations

from typing import Any

POINTER_COLUMNS = "chunk_key, url, anchor, page_title, heading_path, library, kind, source_link"

DENSE_SQL = f"""
select {POINTER_COLUMNS} from chunks
{{where}}
order by embedding <=> %(vector)s::vector
limit %(pool)s
"""

KEYWORD_SQL = f"""
select {POINTER_COLUMNS} from chunks
where tsv @@ plainto_tsquery('english', %(query)s) {{extra}}
order by ts_rank(tsv, plainto_tsquery('english', %(query)s)) desc
limit %(pool)s
"""


def rrf_merge(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(item) = Σ 1/(k + rank) over all rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def _rows_to_pointers(rows: list[tuple]) -> dict[str, dict[str, Any]]:
    keys = POINTER_COLUMNS.replace(" ", "").split(",")
    return {row[0]: dict(zip(keys, row, strict=True)) for row in rows}


def retrieve(
    query: str,
    k: int = 8,
    library: str | None = None,
    conn=None,
    embed_fn=None,
    pool: int = 20,
) -> list[dict[str, Any]]:
    """Top-k pointers for a query: dense (pgvector) + keyword (tsvector) via RRF.

    `pool` candidates are taken from each modality before fusion — the classic
    setup where keyword search rescues exact symbol names (e.g.
    `scaled_dot_product_attention`) that dense similarity alone misses.
    """
    from index.db import connect
    from index.embed import embed_query

    embed_fn = embed_fn or embed_query
    own_conn = conn is None
    if own_conn:
        conn = connect()
    try:
        params: dict[str, Any] = {"pool": pool, "query": query}
        where, extra = "", ""
        if library:
            where, extra = "where library = %(library)s", "and library = %(library)s"
            params["library"] = library

        params["vector"] = str(embed_fn(query))
        dense_rows = conn.execute(DENSE_SQL.format(where=where), params).fetchall()
        keyword_rows = conn.execute(KEYWORD_SQL.format(extra=extra), params).fetchall()
    finally:
        if own_conn:
            conn.close()

    pointers = _rows_to_pointers(dense_rows) | _rows_to_pointers(keyword_rows)
    scores = rrf_merge(
        [[r[0] for r in dense_rows], [r[0] for r in keyword_rows]]
    )
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [pointers[chunk_key] for chunk_key in top]
