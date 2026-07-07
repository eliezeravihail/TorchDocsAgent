"""Hybrid retrieval over the chunks index: dense + keyword, RRF-merged.

Returns POINTERS (url, anchor, heading path, ...) — never content. The
caller hydrates content from the snapshot (index/hydrate.py). This is the
engine behind the agent's `search_docs` tool.
"""

from __future__ import annotations

import re
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

# exact-symbol channel: for API-name queries the reference page is short and
# loses to verbose tutorials on both dense and keyword — match the symbol in the
# url/title/heading and prefer api-reference pages, so RRF lifts the real def
SYMBOL_SQL = f"""
select {POINTER_COLUMNS} from chunks
where (url ilike %(sym)s or page_title ilike %(sym)s or heading_path ilike %(sym)s) {{extra}}
order by (kind = 'api') desc, length(url) asc
limit %(pool)s
"""

# a symbol-ish token: dotted/underscored identifier (torch.nn.functional.sdpa,
# scaled_dot_product_attention). Bare words like "SGD" go through dense+keyword.
_SYMBOL_TOKEN = re.compile(r"[A-Za-z_][\w.]*[._][\w.]*[A-Za-z0-9_]")


def extract_symbol(query: str) -> str | None:
    """Longest dotted/underscored identifier in the query, if any."""
    matches = _SYMBOL_TOKEN.findall(query)
    return max(matches, key=len) if matches else None


def is_exact_api(pointer: dict, symbol: str) -> bool:
    """True if the pointer is the API-reference page for exactly this symbol.

    url .../generated/torch.nn.functional.scaled_dot_product_attention.html
    matches symbol 'scaled_dot_product_attention' (suffix) or the full path.
    """
    if pointer.get("kind") != "api":
        return False
    stem = pointer["url"].rsplit("/", 1)[-1].removesuffix(".html")
    return stem == symbol or stem.endswith("." + symbol)


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


def top_distance(query: str, conn=None, embed_fn=None) -> float | None:
    """Smallest cosine distance between the query and any chunk (0 = identical).

    A topicality signal for the input guard: the dense channel always returns
    the nearest chunks, so "did it return rows" can't tell on-topic from
    off-topic — but an off-topic query's *nearest* chunk is still far. Returns
    None if the index is empty. pgvector `<=>` is cosine distance in [0, 2].
    """
    from index.db import connect
    from index.embed import embed_query

    embed_fn = embed_fn or embed_query
    own_conn = conn is None
    if own_conn:
        conn = connect()
    try:
        row = conn.execute(
            "select embedding <=> %(vector)s::vector as dist from chunks "
            "order by dist limit 1",
            {"vector": str(embed_fn(query))},
        ).fetchone()
    finally:
        if own_conn:
            conn.close()
    return float(row[0]) if row else None


def retrieve(
    query: str,
    k: int = 8,
    library: str | None = None,
    conn=None,
    embed_fn=None,
    pool: int = 20,
    debug: bool = False,
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

        symbol_rows: list[tuple] = []
        symbol = extract_symbol(query)
        if symbol:
            params["sym"] = f"%{symbol}%"
            symbol_rows = conn.execute(SYMBOL_SQL.format(extra=extra), params).fetchall()
    finally:
        if own_conn:
            conn.close()

    channels = [("dense", dense_rows), ("keyword", keyword_rows), ("symbol", symbol_rows)]
    if debug:
        for name, rows in channels:
            print(f"[debug] {name}: {len(rows)} candidates")
            for row in rows[:3]:
                print(f"[debug]   {row[4]} | {row[1]}")

    pointers: dict[str, dict[str, Any]] = {}
    for _, rows in channels:
        pointers |= _rows_to_pointers(rows)
    scores = rrf_merge([[r[0] for r in rows] for _, rows in channels])
    ranked = sorted(scores, key=scores.get, reverse=True)

    # exact API lookup: if the user typed a precise symbol and its reference
    # page was found, pin it first — the docs-search behavior users expect
    if symbol:
        exact = next((ck for ck in ranked if is_exact_api(pointers[ck], symbol)), None)
        if exact:
            ranked = [exact] + [ck for ck in ranked if ck != exact]

    return [pointers[chunk_key] for chunk_key in ranked[:k]]
