"""Hybrid retrieval over the chunks index: per-kind pools, seam-free by design.

Each content kind (api / tutorial / guide) gets its own pool — dense +
keyword, RRF-merged WITHIN the kind — so verbose tutorials can never crowd
reference pages out of the results (and vice versa). A per-pool relevance
threshold drops candidates far from that kind's own best hit, pools are
interleaved strongest-first into the top-k, and the answering model judges
what is actually relevant.
This replaced a global-RRF design whose crowding fixes (a reference channel,
reserved seats, distance gates) traded one benchmark regression for another.
A cross-encoder rerank (index/rerank.py) then reorders a wide slate of the
fused candidates into the final top-k — precision after candidate generation.

Returns POINTERS (url, anchor, heading path, ...) — never content. The
caller hydrates content from the snapshot (index/hydrate.py). This is the
engine behind the agent's `search_docs` tool.
"""

from __future__ import annotations

import re
from typing import Any

# `content` rides along in the same query retrieval already runs, so the answer
# path hydrates each section from it with zero extra round-trips or live fetches
POINTER_COLUMNS = (
    "chunk_key, url, anchor, page_title, heading_path, library, kind, source_link, "
    "part, content"
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

# the content spaces, each searched as its own pool (see ingest's page_kind)
KINDS = ("api", "tutorial", "guide")

# a candidate whose cosine distance exceeds the best hit's by more than this
# is junk that would only dilute the answer context — dropped before
# interleaving. Keyword-only hits (no distance) are kept: an exact lexical
# match on a rare term is a signal of its own.
RELEVANCE_GAP = 0.15

# HNSW is approximate AND pgvector applies WHERE filters *after* the index
# scan: with the default ef_search=40, a `kind='api'` query first collects the
# ~40 globally-nearest chunks (mostly tutorials for a descriptive question) and
# only then filters — the api page can be discarded before the filter ever sees
# it. Diagnosed on the live index: torch.optim.SGD sat at TRUE dense rank 7
# within api, yet was absent from the returned top-20. A wider candidate scan
# fixes exactly this; ~2ms extra per query on a 7K-chunk index.
HNSW_EF_SEARCH = 150

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
    rerank_fn=None,
) -> list[dict[str, Any]]:
    """Top-k pointers for a query, drawn from per-kind pools.

    Each kind (api / tutorial / guide) is searched separately — dense
    (pgvector) + keyword (tsvector), RRF-merged within the pool, `pool`
    candidates per modality. Keyword search rescues exact symbol names
    (e.g. `scaled_dot_product_attention`) that dense similarity misses.

    Candidates farther than RELEVANCE_GAP from the best hit are dropped, then
    the pools are interleaved (strongest pool first) into a wide slate. A
    cross-encoder rerank (index/rerank.py — the precision stage; kill switch
    TORCHDOCS_RERANK) reorders the slate into the top-k; when it is off the
    slate's first k are the top-k unchanged. An exact-symbol match is pinned
    first, docs-search style.

    `kind` restricts the search to one content space — the agent's planner
    sets it when it decides the question needs the reference catalog rather
    than tutorial prose. `rerank_fn` (tests) replaces the real reranker and
    forces the rerank path on.
    """
    from contextlib import nullcontext

    from index.db import get_pool
    from index.embed import embed_query

    embed_fn = embed_fn or embed_query
    # No caller connection → borrow one from the shared pool for these reads and
    # return it on exit. An injected conn (tests, the batch build) is used as-is
    # and left open for its owner to manage.
    ctx = nullcontext(conn) if conn is not None else get_pool().connection()
    kinds = (kind,) if kind else KINDS
    pools: list[tuple[str, list[str]]] = []  # (pool name, ranked chunk_keys)
    pointers: dict[str, dict[str, Any]] = {}
    dists: dict[str, float] = {}

    with ctx as conn:
        conn.execute(f"set hnsw.ef_search = {HNSW_EF_SEARCH:d}")
        base: dict[str, Any] = {"pool": pool, "query": query, "vector": str(embed_fn(query))}
        conditions = ["kind = %(kind)s"]
        if library:
            conditions.append("library = %(library)s")
            base["library"] = library
        where = "where " + " and ".join(conditions)
        extra = "".join(f"and {c} " for c in conditions)

        for kd in kinds:
            params = {**base, "kind": kd}
            dense_rows = conn.execute(DENSE_SQL.format(where=where), params).fetchall()
            keyword_rows = conn.execute(KEYWORD_SQL.format(extra=extra), params).fetchall()
            pointers |= _rows_to_pointers(dense_rows) | _rows_to_pointers(keyword_rows)
            dists |= _collect_distances([dense_rows])
            scores = rrf_merge(
                [[r[0] for r in dense_rows], [r[0] for r in keyword_rows]]
            )
            pools.append((kd, sorted(scores, key=scores.get, reverse=True)))

        symbol = extract_symbol(query)
        if symbol:
            # escape ILIKE wildcards so a literal % or _ in the query matches
            # itself instead of acting as a pattern (backslash is the default
            # escape character in Postgres LIKE/ILIKE)
            escaped = symbol.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            sym_extra = "".join(
                f"and {c} " for c in conditions if not c.startswith("kind") or kind
            )
            params = {**base, "sym": f"%{escaped}%", "kind": kind}
            symbol_rows = conn.execute(SYMBOL_SQL.format(extra=sym_extra), params).fetchall()
            pointers |= _rows_to_pointers(symbol_rows)
            # the symbol pool leads the interleave: an explicit identifier is
            # the strongest possible signal of what the user wants
            pools.insert(0, ("symbol", [r[0] for r in symbol_rows]))

    if debug:
        for name, ranked in pools:
            print(f"[debug] pool {name}: {len(ranked)} candidates")
            for ck in ranked[:3]:
                print(f"[debug]   {pointers[ck]['heading_path']} | {pointers[ck]['url']}")

    from index import rerank as rerank_mod

    # rerank reads a slate wider than k; round-robin order is prefix-stable, so
    # with reranking off (or failing open) the first k match today's behavior
    use_rerank = rerank_fn is not None or rerank_mod.enabled()
    slate = max(k, rerank_mod.RERANK_SLATE) if use_rerank else k
    ranked = _interleave_pools(pools, dists, slate)

    if use_rerank and len(ranked) > 1:
        rerank_fn = rerank_fn or rerank_mod.rerank
        ranked = [p["chunk_key"] for p in rerank_fn(query, [pointers[ck] for ck in ranked], k=k)]

    # exact API lookup: if the user typed a precise symbol and its reference
    # page was found, pin it first — the docs-search behavior users expect
    # (an explicit identifier outranks even the cross-encoder's judgment)
    if symbol:
        exact = next(
            (ck for kd, pl in pools for ck in pl if is_exact_api(pointers[ck], symbol)),
            None,
        )
        if exact:
            ranked = [exact] + [ck for ck in ranked if ck != exact]

    return [pointers[chunk_key] for chunk_key in ranked[:k]]


def _interleave_pools(
    pools: list[tuple[str, list[str]]], dists: dict[str, float], k: int
) -> list[str]:
    """Merge per-kind rankings into one top-k, round-robin, strongest pool first.

    The relevance threshold runs here, PER POOL: within each kind, candidates
    farther than RELEVANCE_GAP from that kind's own nearest hit are dropped
    (keyword-only candidates carry no distance and are kept). Gapping per pool,
    not against a global best, is the whole point of per-kind pools — a close
    tutorial must not set a threshold that filters the entire api pool out, or
    a descriptive question that a tutorial answers first would never surface its
    reference page at all. Round-robin then guarantees every surviving space a
    share of the k seats; ordering pools by their best distance keeps the most
    relevant kind in front, so a tutorial-shaped question still leads with
    tutorials. The answering model does the final relevance judgment.
    """

    def pool_strength(ranked: list[str]) -> float:
        known = [dists[ck] for ck in ranked if ck in dists]
        return min(known) if known else float("inf")

    def keep_within_gap(ranked: list[str]) -> list[str]:
        best = pool_strength(ranked)  # this pool's own nearest hit
        if best == float("inf"):  # keyword-only pool — no distances to gate on
            return ranked
        return [ck for ck in ranked if (d := dists.get(ck)) is None or d <= best + RELEVANCE_GAP]

    filtered = [(name, keep_within_gap(ranked)) for name, ranked in pools]
    # the symbol pool (if present) was inserted first and stays first; the
    # kind pools compete on the strength of their best surviving candidate
    lead = [p for p in filtered if p[0] == "symbol"]
    rest = sorted((p for p in filtered if p[0] != "symbol"), key=lambda p: pool_strength(p[1]))
    ordered = [ranked for _, ranked in lead + rest]

    out: list[str] = []
    seen: set[str] = set()
    positions = [0] * len(ordered)
    progressed = True
    while len(out) < k and progressed:
        progressed = False
        for i, ranked in enumerate(ordered):
            while positions[i] < len(ranked) and ranked[positions[i]] in seen:
                positions[i] += 1
            if positions[i] < len(ranked):
                ck = ranked[positions[i]]
                positions[i] += 1
                seen.add(ck)
                out.append(ck)
                progressed = True
                if len(out) == k:
                    break
    return out
