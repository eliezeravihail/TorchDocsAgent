from index.retrieve import extract_symbol, retrieve, rrf_merge


def test_rrf_prefers_items_ranked_in_both():
    scores = rrf_merge([["a", "b", "c"], ["b", "d"]])
    assert scores["b"] > scores["a"] > scores["c"]
    assert "d" in scores


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
    return (key, url, "anchor", "title", heading, "core", "api", "")


def test_retrieve_merges_dense_and_keyword():
    dense = [_row("d1"), _row("both"), _row("d3")]
    keyword = [_row("both"), _row("k2")]
    conn = FakeConn([dense, keyword])
    results = retrieve("early stopping", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    keys = [r["chunk_key"] for r in results]
    assert keys[0] == "both"  # ranked by both modalities → wins RRF
    assert len(keys) == 3
    assert set(keys) <= {"d1", "both", "d3", "k2"}


def test_retrieve_library_filter_lands_in_sql():
    conn = FakeConn([[], []])
    retrieve("q", library="vision", conn=conn, embed_fn=lambda q: [0.0] * 384)
    dense_sql, dense_params = conn.queries[0]
    assert "where library" in dense_sql
    assert dense_params["library"] == "vision"


def test_symbol_query_adds_third_channel_and_wins():
    dense = [_row("d1"), _row("d2")]
    keyword = [_row("k1")]
    symbol = [_row("api")]  # the exact API-reference page, only in the symbol channel
    conn = FakeConn([dense, keyword, symbol])
    results = retrieve(
        "scaled_dot_product_attention", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384
    )
    assert len(conn.queries) == 3  # dense + keyword + symbol
    assert conn.queries[2][1]["sym"] == "%scaled_dot_product_attention%"
    assert "api" in [r["chunk_key"] for r in results]


def test_non_symbol_query_skips_symbol_channel():
    conn = FakeConn([[_row("d1")], [_row("k1")]])
    retrieve("how to train a model", k=3, conn=conn, embed_fn=lambda q: [0.0] * 384)
    assert len(conn.queries) == 2  # no symbol channel
