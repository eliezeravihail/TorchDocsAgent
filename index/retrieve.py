"""Hybrid retrieval over the chunks index: dense + keyword, RRF-merged.

Returns POINTERS (url, anchor, heading path, ...) — never content. The
caller hydrates content from the snapshot (index/hydrate.py). This is the
engine behind the agent's `search_docs` tool.
"""

from __future__ import annotations

import re
from typing import Any

POINTER_COLUMNS = (
    "chunk_key, url, anchor, page_title, heading_path, library, kind, source_link, part"
)

DENSE_SQL = f"""
select {POINTER_COLUMNS}, embedding <=> %(vector)s::vector as dist from chunks
{{where}}
order by dist
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

# reference channel: dense search restricted to kind='api'. Catalog/source
# questions with no explicit symbol ("what loss functions exist?") drown in
# tutorial prose on the open channels — tutorials say "loss function" a dozen
# times, the reference page once. This channel guarantees reference candidates
# reach the RRF pool at all; _ensure_api_slots below guarantees the best of
# them survive into the returned top-k.
API_DENSE_SQL = f"""
select {POINTER_COLUMNS}, embedding <=> %(vector)s::vector as dist from chunks
where kind = 'api' {{extra}}
order by dist
limit %(pool)s
"""

# Reservation limits, tuned on the retrieval benchmark (2026-07-08): with 2
# blind seats, promoted-but-irrelevant reference pages displaced a real hit at
# rank 7 (q10 regressed 1.00→0.00) and junk (torchaudio RNNT for an SGD
# question) got promoted. One seat, and only for a candidate whose cosine
# distance is within API_SLOT_MAX_GAP of the best result — a reserved seat for
# QUALIFIED references, not for whatever the api channel happened to rank.
MIN_API_IN_TOP_K = 1
API_SLOT_MAX_GAP = 0.10

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
    keys = POINTER_COLUMNS.replace(" ", "").replace("\n", "").split(",")
    # dense-channel rows carry a trailing `dist` column beyond the pointer
    # fields — slice it off here; retrieve() collects it separately
    return {row[0]: dict(zip(keys, row[: len(keys)], strict=True)) for row in rows}


def _collect_distances(row_sets: list[list[tuple]]) -> dict[str, float]:
    """chunk_key → best cosine distance, from rows that carry a dist column."""
    n_pointer_cols = len(POINTER_COLUMNS.replace(" ", "").replace("\n", "").split(","))
    dists: dict[str, float] = {}
    for rows in row_sets:
        for row in rows:
            if len(row) > n_pointer_cols:
                d = float(row[n_pointer_cols])
                dists[row[0]] = min(dists.get(row[0], d), d)
    return dists


def top_distance(query: str, conn=None, embed_fn=None) -> float | None:
    """Smallest cosine distance between the query and any chunk (0 = identical).

    A topicality signal for the input guard: the dense channel always returns
    the nearest chunks, so "did it return rows" can't tell on-topic from
    off-topic — but an off-topic query's *nearest* chunk is still far. Returns
    None if the index is empty. pgvector `<=>` is cosine distance in [0, 2].
    """
    from contextlib import nullcontext

    from index.db import get_pool
    from index.embed import embed_query

    embed_fn = embed_fn or embed_query
    # Same pooled access as retrieve(): this runs on every guarded question, so
    # a raw connect() here would pay a TLS handshake per request and hold
    # connections outside the pool's cap.
    ctx = nullcontext(conn) if conn is not None else get_pool().connection()
    with ctx as conn:
        row = conn.execute(
            "select embedding <=> %(vector)s::vector as dist from chunks "
            "order by dist limit 1",
            {"vector": str(embed_fn(query))},
        ).fetchone()
    return float(row[0]) if row else None


def retrieve(
    query: str,
    k: int = 8,
    library: str | None = None,
    conn=None,
    embed_fn=None,
    pool: int = 20,
    debug: bool = False,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """Top-k pointers for a query: dense (pgvector) + keyword (tsvector) via RRF.

    `pool` candidates are taken from each modality before fusion — the classic
    setup where keyword search rescues exact symbol names (e.g.
    `scaled_dot_product_attention`) that dense similarity alone misses.

    `kind` restricts the search to one content space ('api' / 'tutorial' /
    'guide'). The agent's planner sets it when it decides the question needs
    the reference catalog rather than tutorial prose — the caller's judgment
    replaces the api-channel/reservation heuristics, which are skipped.
    """
    from contextlib import nullcontext

    from index.db import get_pool
    from index.embed import embed_query

    embed_fn = embed_fn or embed_query
    # No caller connection → borrow one from the shared pool for these reads and
    # return it on exit. An injected conn (tests, the batch build) is used as-is
    # and left open for its owner to manage.
    ctx = nullcontext(conn) if conn is not None else get_pool().connection()
    with ctx as conn:
        params: dict[str, Any] = {"pool": pool, "query": query}
        conditions = []
        if library:
            conditions.append("library = %(library)s")
            params["library"] = library
        if kind:
            conditions.append("kind = %(kind)s")
            params["kind"] = kind
        where = ("where " + " and ".join(conditions)) if conditions else ""
        extra = "".join(f"and {c} " for c in conditions)

        params["vector"] = str(embed_fn(query))
        dense_rows = conn.execute(DENSE_SQL.format(where=where), params).fetchall()
        keyword_rows = conn.execute(KEYWORD_SQL.format(extra=extra), params).fetchall()
        # an explicit kind makes the reference channel redundant (kind='api')
        # or contradictory (kind='tutorial') — the caller chose the space
        api_rows: list[tuple] = []
        if kind is None:
            api_rows = conn.execute(API_DENSE_SQL.format(extra=extra), params).fetchall()

        symbol_rows: list[tuple] = []
        symbol = extract_symbol(query)
        if symbol:
            # escape ILIKE wildcards so a literal % or _ in the query matches
            # itself instead of acting as a pattern (backslash is the default
            # escape character in Postgres LIKE/ILIKE)
            escaped = symbol.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            params["sym"] = f"%{escaped}%"
            symbol_rows = conn.execute(SYMBOL_SQL.format(extra=extra), params).fetchall()

    channels = [
        ("dense", dense_rows),
        ("keyword", keyword_rows),
        ("api", api_rows),
        ("symbol", symbol_rows),
    ]
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

    if kind is None:  # with an explicit kind the caller already chose the space
        dists = _collect_distances([dense_rows, api_rows])
        ranked = _ensure_api_slots(ranked, pointers, k, dists)
    return [pointers[chunk_key] for chunk_key in ranked[:k]]


def _ensure_api_slots(
    ranked: list[str],
    pointers: dict[str, dict],
    k: int,
    dists: dict[str, float] | None = None,
    min_api: int = MIN_API_IN_TOP_K,
) -> list[str]:
    """Guarantee QUALIFIED reference pages a seat in the top-k.

    Tutorials mention a topic many times and rank in every open channel, so a
    non-symbol question can fill the whole top-k with tutorial prose while the
    reference page — the canonical answer — sits just below the cut. When
    fewer than min_api reference (kind='api') pointers made the top-k, promote
    the best-ranked reference candidates over the lowest-ranked tutorials.

    Qualification: a candidate is promoted only when its cosine distance is
    within API_SLOT_MAX_GAP of the best result overall — a reference page that
    is nearly as relevant as the top hit deserves a seat; one that merely won
    the api channel by default does not (benchmark run 3: blind promotion put
    torchaudio's RNNT page on an SGD question and displaced a real hit).
    When no distance information is available (injected test rows), the gate
    is open.
    """
    dists = dists or {}
    best = min(dists.values()) if dists else None

    def qualified(ck: str) -> bool:
        if best is None:
            return True
        d = dists.get(ck)
        return d is not None and d <= best + API_SLOT_MAX_GAP

    top = ranked[:k]
    have = sum(1 for ck in top if pointers[ck].get("kind") == "api")
    extras = [
        ck for ck in ranked[k:] if pointers[ck].get("kind") == "api" and qualified(ck)
    ]
    need = min(min_api - have, len(extras))
    if need <= 0:
        return ranked
    # drop the lowest-ranked non-reference entries to make room, keep order
    to_drop: list[str] = []
    for ck in reversed(top):
        if len(to_drop) == need:
            break
        if pointers[ck].get("kind") != "api":
            to_drop.append(ck)
    new_top = [ck for ck in top if ck not in to_drop] + extras[:need]
    return new_top + [ck for ck in ranked if ck not in new_top]
