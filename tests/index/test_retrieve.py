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

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        rows = self._results.pop(0)
        from types import SimpleNamespace

        return SimpleNamespace(fetchall=lambda: rows)


def _row(key, url="https://docs.pytorch.org/docs/stable/x.html", heading="H"):
    return (key, url, "anchor", "title", heading, "core", "api", "", 0)


def test_retrieve_merges_dense_and_keyword():
    dense = [_row("d1"), _row("both"), _row("d3")]
    keyword = [_row("both"), _row("k2")]
    conn = FakeConn([dense, keyword, []])  # third set: the api channel
    results = retrieve("early stopping", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    keys = [r["chunk_key"] for r in results]
    assert keys[0] == "both"  # ranked by both modalities → wins RRF
    assert len(keys) == 3
    assert set(keys) <= {"d1", "both", "d3", "k2"}


def test_retrieve_library_filter_lands_in_sql():
    conn = FakeConn([[], [], []])
    retrieve("q", library="vision", conn=conn, embed_fn=lambda q: [0.0] * 384)
    dense_sql, dense_params = conn.queries[0]
    assert "where library" in dense_sql
    assert dense_params["library"] == "vision"
    api_sql, _ = conn.queries[2]
    assert "kind = 'api'" in api_sql and "and library" in api_sql


def test_symbol_query_adds_third_channel_and_wins():
    dense = [_row("d1"), _row("d2")]
    keyword = [_row("k1")]
    symbol = [_row("api")]  # the exact API-reference page, only in the symbol channel
    conn = FakeConn([dense, keyword, [], symbol])
    results = retrieve(
        "scaled_dot_product_attention", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384
    )
    assert len(conn.queries) == 4  # dense + keyword + api + symbol
    # ILIKE wildcards in the symbol are escaped: \_ matches a literal _
    assert conn.queries[3][1]["sym"] == r"%scaled\_dot\_product\_attention%"
    assert "api" in [r["chunk_key"] for r in results]


def test_non_symbol_query_skips_symbol_channel():
    conn = FakeConn([[_row("d1")], [_row("k1")], []])
    retrieve("how to train a model", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert len(conn.queries) == 3  # dense + keyword + api; no symbol channel


def test_retrieve_without_conn_borrows_and_returns_pool_connection(monkeypatch):
    # with no caller conn, retrieve() must take a connection from the shared pool
    # and hand it back (context-manager exit) so the pool isn't drained under load
    conn = FakeConn([[_row("d1")], [_row("k1")], []])
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

    assert {r["chunk_key"] for r in results} == {"d1", "k1"}
    assert returned["n"] == 1  # connection was returned to the pool exactly once


def _api_row(key, symbol):
    url = f"https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.{symbol}.html"
    return (key, url, "", "", "", "core", "api", "", 0)


def test_exact_api_page_pinned_first():
    # tutorials dominate RRF (appear in every channel); the API page appears
    # only in the symbol channel yet must be pinned #1 for an exact symbol query
    dense = [_row("tut1"), _row("tut2")]
    keyword = [_row("tut1")]
    symbol = [_row("tut1"), _api_row("apipage", "scaled_dot_product_attention")]
    conn = FakeConn([dense, keyword, [], symbol])
    results = retrieve(
        "scaled_dot_product_attention", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384
    )
    assert results[0]["chunk_key"] == "apipage"


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


# --- reference representation (the q06/q13 failure mode) --------------------


def _tut_row(key):
    url = f"https://docs.pytorch.org/tutorials/beginner/{key}.html"
    return (key, url, "", "", "Loss Functions", "tutorials", "tutorial", "", 0)


def _ref_row(key):
    url = f"https://docs.pytorch.org/docs/stable/generated/torch.nn.{key}.html"
    return (key, url, "", "", "", "core", "api", "", 0)


def test_reference_pages_get_seats_when_tutorials_fill_the_topk():
    # a catalog question with no symbol: tutorials rank in BOTH open channels
    # and would fill all of top-k; the api channel + slot reservation must put
    # the best reference pages in anyway
    tutorials = [_tut_row(f"tut{i}") for i in range(8)]
    api = [_ref_row("CrossEntropyLoss"), _ref_row("NLLLoss"), _ref_row("MSELoss")]
    conn = FakeConn([tutorials, tutorials, api])
    results = retrieve(
        "what loss functions exist for classification", k=4, conn=conn,
        embed_fn=lambda q: [0.0] * 384,
    )
    keys = [r["chunk_key"] for r in results]
    assert len(keys) == 4
    # the two best-ranked reference pages made it in, in channel order
    assert "CrossEntropyLoss" in keys and "NLLLoss" in keys
    # and the top tutorials survived — reservation replaces the tail, not the head
    assert keys[0] == "tut0"


def test_no_reservation_when_references_already_present():
    # references ranking naturally in the top-k → nothing is reshuffled
    dense = [_ref_row("SGD"), _tut_row("tut1"), _ref_row("Adam")]
    conn = FakeConn([dense, [], []])
    results = retrieve("optimizers", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert [r["chunk_key"] for r in results] == ["SGD", "tut1", "Adam"]


def test_reservation_never_invents_candidates():
    # nothing but tutorials anywhere → top-k stays all-tutorial (no crash)
    tutorials = [_tut_row(f"tut{i}") for i in range(5)]
    conn = FakeConn([tutorials, [], []])
    results = retrieve("how to train", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert all(r["kind"] == "tutorial" for r in results)
