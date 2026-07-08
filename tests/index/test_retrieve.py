from types import SimpleNamespace

from index.retrieve import extract_symbol, retrieve, rrf_merge, top_distance


def test_rrf_prefers_items_ranked_in_both():
    scores = rrf_merge([["a", "b", "c"], ["b", "d"]])
    assert scores["b"] > scores["a"] > scores["c"]
    assert "d" in scores


def _distance_conn(row):
    return SimpleNamespace(execute=lambda sql, params=None: SimpleNamespace(fetchone=lambda: row))


def _emb(q):
    return [0.0] * 384


def test_top_distance_returns_best_cosine_distance():
    assert top_distance("how do I use SGD?", conn=_distance_conn((0.42,)), embed_fn=_emb) == 0.42


def test_top_distance_none_on_empty_index():
    assert top_distance("anything", conn=_distance_conn(None), embed_fn=_emb) is None


def test_extract_symbol():
    assert extract_symbol("scaled_dot_product_attention") == "scaled_dot_product_attention"
    assert extract_symbol("how do I use torch.optim.SGD?") == "torch.optim.SGD"
    assert extract_symbol("how do I train a network") is None
    assert extract_symbol("what is SGD") is None  # bare word, no dot/underscore


class FakeConn:
    """Returns queued row lists for successive execute() calls."""

    def __init__(self, result_sets):
        self._results = list(result_sets)
        self.queries = []
        self.settings = []  # session SETs (e.g. hnsw.ef_search), kept separate

    def execute(self, sql, params=None):
        if sql.lstrip().lower().startswith("set "):
            self.settings.append(sql)
            return None
        self.queries.append((sql, params))
        rows = self._results.pop(0)
        from types import SimpleNamespace

        return SimpleNamespace(fetchall=lambda: rows)


def _row(key, kind="api", heading="H", url=None, dist=None):
    """A fake DB row; dense rows carry a trailing dist column, keyword rows don't."""
    if url is None:
        base = "docs/stable/generated" if kind == "api" else f"{kind}s/beginner"
        url = f"https://docs.pytorch.org/{base}/{key}.html"
    row = (key, url, "anchor", "title", heading, "core", kind, "", 0)
    return row if dist is None else (*row, dist)


def _no_pools(n=6):
    """Empty result sets for the 3 kind-pools × (dense, keyword)."""
    return [[] for _ in range(n)]


def test_is_exact_api():
    from index.retrieve import is_exact_api

    ptr = {
        "kind": "api",
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html",
    }
    assert is_exact_api(ptr, "SGD")
    assert is_exact_api(ptr, "torch.optim.SGD")
    assert not is_exact_api(ptr, "Adam")
    assert not is_exact_api({**ptr, "kind": "tutorial"}, "SGD")


# --- per-kind pools ----------------------------------------------------------
# query order without an explicit kind: (dense, keyword) × (api, tutorial,
# guide), then the symbol query when the query contains an identifier


def test_within_pool_rrf_merges_dense_and_keyword():
    dense = [_row("d1"), _row("both"), _row("d3")]
    keyword = [_row("both"), _row("k2")]
    conn = FakeConn([dense, keyword, [], [], [], []])
    results = retrieve("early stopping", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    keys = [r["chunk_key"] for r in results]
    assert keys[0] == "both"  # ranked by both modalities → wins RRF within the pool
    assert len(keys) == 3


def test_pools_are_interleaved_so_no_kind_crowds_out_another():
    # the q06/q10 seesaw: previously whichever kind dominated the global
    # ranking took ALL k seats; now every space with candidates is represented
    refs = [_row(f"ref{i}", kind="api") for i in range(8)]
    tuts = [_row(f"tut{i}", kind="tutorial") for i in range(8)]
    conn = FakeConn([refs, [], tuts, [], [], []])
    results = retrieve("loss functions", k=6, conn=conn, embed_fn=lambda q: [0.0] * 384)
    kinds = [r["kind"] for r in results]
    assert kinds.count("api") == 3 and kinds.count("tutorial") == 3
    # within each pool, rank order is preserved
    assert [r["chunk_key"] for r in results if r["kind"] == "api"] == ["ref0", "ref1", "ref2"]


def test_strongest_pool_leads_the_interleave():
    # a tutorial-shaped question: the tutorial pool's best hit is closest, so
    # tutorials come first — diversity must not cost the natural #1
    refs = [_row("ref0", kind="api", dist=0.30)]
    tuts = [_row("tut0", kind="tutorial", dist=0.18), _row("tut1", kind="tutorial", dist=0.20)]
    conn = FakeConn([refs, [], tuts, [], [], []])
    results = retrieve("fine tuning walkthrough", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert [r["chunk_key"] for r in results] == ["tut0", "ref0", "tut1"]


def test_relevance_threshold_drops_far_candidates_within_a_pool():
    # the gap runs PER POOL: an api page far from the api pool's OWN best is
    # dropped; the pool's near cluster is kept; a keyword-only hit (no distance)
    # is kept — exact lexical match on a rare term is a signal of its own
    api = [_row("near_ref", kind="api", dist=0.30), _row("far_ref", kind="api", dist=0.55)]
    tuts = [_row("tut0", kind="tutorial", dist=0.20)]
    kw_only = [_row("rare-term-hit", kind="guide")]
    conn = FakeConn([api, [], tuts, [], [], kw_only])
    results = retrieve("update weights sgd", k=4, conn=conn, embed_fn=lambda q: [0.0] * 384)
    keys = [r["chunk_key"] for r in results]
    assert "far_ref" not in keys  # 0.55 > 0.30 + 0.15, its own pool's outlier
    assert "near_ref" in keys and "tut0" in keys and "rare-term-hit" in keys


def test_close_tutorial_does_not_empty_the_api_pool():
    # regression for the global-best gap bug: a close tutorial (0.18) used to set
    # a threshold that filtered out EVERY api candidate (0.40), so a descriptive
    # question a tutorial answered first never surfaced its reference page at all.
    # Per-pool gapping keeps the api pool's own near cluster regardless.
    api = [_row("ref0", kind="api", dist=0.40)]
    tuts = [_row("tut0", kind="tutorial", dist=0.18)]
    conn = FakeConn([api, [], tuts, [], [], []])
    results = retrieve("fully connected layer", k=4, conn=conn, embed_fn=lambda q: [0.0] * 384)
    keys = [r["chunk_key"] for r in results]
    assert "ref0" in keys  # 0.40 fails a global 0.18+0.15 gap, but its pool keeps it
    assert keys[0] == "tut0"  # the closer pool still leads the interleave


def test_library_filter_lands_in_every_pool_query():
    conn = FakeConn(_no_pools())
    retrieve("q", library="vision", conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert len(conn.queries) == 6
    kinds_queried = [p["kind"] for _, p in conn.queries[::2]]  # dense query of each pool
    assert kinds_queried == ["api", "tutorial", "guide"]
    for sql, params in conn.queries:
        assert "library = %(library)s" in sql and params["library"] == "vision"


def test_non_symbol_query_runs_no_symbol_query():
    conn = FakeConn(_no_pools())
    retrieve("how to train a model", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert len(conn.queries) == 6


def test_dense_search_widens_the_hnsw_candidate_scan():
    # pgvector filters AFTER the approximate index scan: at the default
    # ef_search=40 a kind-filtered query can drop a page that is truly rank-7
    # within its kind (measured live: torch.optim.SGD). The session must widen
    # the scan before any dense query runs.
    from index.retrieve import HNSW_EF_SEARCH

    conn = FakeConn(_no_pools())
    retrieve("sgd update rule", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert conn.settings and f"ef_search = {HNSW_EF_SEARCH}" in conn.settings[0]


def test_symbol_query_escapes_ilike_and_pins_exact_page():
    # tutorials dominate the pools; the exact API page appears only via the
    # symbol query yet must be pinned #1 — the docs-search behavior users expect
    tuts = [_row(f"tut{i}", kind="tutorial") for i in range(3)]
    exact = _row(
        "apipage",
        url="https://docs.pytorch.org/docs/stable/generated/"
        "torch.nn.functional.scaled_dot_product_attention.html",
    )
    conn = FakeConn([[], [], tuts, [], [], [], [tuts[0], exact]])
    results = retrieve(
        "scaled_dot_product_attention", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384
    )
    assert len(conn.queries) == 7  # 6 pool queries + the symbol query
    # ILIKE wildcards in the symbol are escaped: \_ matches a literal _
    assert conn.queries[6][1]["sym"] == r"%scaled\_dot\_product\_attention%"
    assert results[0]["chunk_key"] == "apipage"


def test_explicit_kind_searches_only_that_pool():
    conn = FakeConn([[_row("SGD")], []])
    results = retrieve(
        "what optimizers exist", k=3, kind="api", conn=conn, embed_fn=lambda q: [0.0] * 384
    )
    assert len(conn.queries) == 2  # one pool: dense + keyword
    _, params = conn.queries[0]
    assert params["kind"] == "api"
    assert [r["chunk_key"] for r in results] == ["SGD"]


def test_kind_and_library_filters_combine():
    conn = FakeConn([[], []])
    retrieve(
        "datasets", kind="api", library="vision", conn=conn, embed_fn=lambda q: [0.0] * 384
    )
    _, params = conn.queries[0]
    assert params["kind"] == "api" and params["library"] == "vision"


def test_retrieve_without_conn_borrows_and_returns_pool_connection(monkeypatch):
    # with no caller conn, retrieve() must take a connection from the shared pool
    # and hand it back (context-manager exit) so the pool isn't drained under load
    conn = FakeConn([[_row("d1")], [], [], [], [], []])
    returned = {"n": 0}

    class FakePoolCtx:
        def __enter__(self):
            return conn

        def __exit__(self, *exc):
            returned["n"] += 1
            return False

    class FakePool:
        def connection(self):
            return FakePoolCtx()

    monkeypatch.setattr("index.db.get_pool", lambda: FakePool())
    results = retrieve("early stopping", k=2, embed_fn=lambda q: [0.0] * 384)

    assert [r["chunk_key"] for r in results] == ["d1"]
    assert returned["n"] == 1  # connection was returned to the pool exactly once
