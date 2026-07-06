from index.retrieve import retrieve, rrf_merge


def test_rrf_prefers_items_ranked_in_both():
    scores = rrf_merge([["a", "b", "c"], ["b", "d"]])
    assert scores["b"] > scores["a"] > scores["c"]
    assert "d" in scores


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
