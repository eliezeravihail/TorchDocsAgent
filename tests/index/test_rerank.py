"""The rerank stage — scoring text, ordering, fail-open, and retrieve() wiring."""

from index.rerank import enabled, rerank, rerank_text
from index.retrieve import retrieve


def _ptr(key, url="https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html",
         title="Linear", heading="Linear", kind="api"):
    return {
        "chunk_key": key, "url": url, "anchor": "", "page_title": title,
        "heading_path": heading, "library": "core", "kind": kind,
        "source_link": "", "part": 0,
    }


def test_enabled_reads_env_per_call(monkeypatch):
    monkeypatch.setenv("TORCHDOCS_RERANK", "0")
    assert not enabled()
    monkeypatch.setenv("TORCHDOCS_RERANK", "1")
    assert enabled()


def test_rerank_text_includes_symbol_title_heading_and_gloss(monkeypatch):
    url = "https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html"
    glosses = {url: "The standard fully-connected layer applying a weight matrix and bias."}
    text = rerank_text(_ptr("a", url=url), glosses)
    assert "torch.nn.Linear" in text  # symbol from the url
    assert "Linear" in text
    assert "fully-connected layer" in text  # the gloss is the semantic payload


def test_rerank_orders_by_score_and_cuts_to_k(monkeypatch):
    monkeypatch.setattr("index.embed.load_glosses", lambda: {})
    ptrs = [_ptr("low"), _ptr("high"), _ptr("mid")]
    scores = {"low": 0.1, "high": 0.9, "mid": 0.5}

    def scorer(pairs):
        # pairs align with ptrs order; look scores up by position
        return [scores[p["chunk_key"]] for p in ptrs]

    out = rerank("q", ptrs, k=2, scorer=scorer)
    assert [p["chunk_key"] for p in out] == ["high", "mid"]


def test_rerank_fails_open_to_the_incoming_order(monkeypatch):
    monkeypatch.setattr("index.embed.load_glosses", lambda: {})

    def broken(pairs):
        raise RuntimeError("model exploded")

    ptrs = [_ptr("first"), _ptr("second"), _ptr("third")]
    out = rerank("q", ptrs, k=2, scorer=broken)
    assert [p["chunk_key"] for p in out] == ["first", "second"]


def test_rerank_single_candidate_skips_scoring():
    # no glosses/model needed — must not even build pairs
    out = rerank("q", [_ptr("only")], k=8, scorer=None)
    assert [p["chunk_key"] for p in out] == ["only"]


# --- retrieve() wiring -------------------------------------------------------

from tests.index.test_retrieve import FakeConn, _row  # noqa: E402 — shared fakes


def test_retrieve_feeds_a_wide_slate_to_the_reranker_and_keeps_its_order():
    refs = [_row(f"ref{i}", kind="api", dist=0.2 + i / 100) for i in range(10)]
    conn = FakeConn([refs, [], [], [], [], []])
    seen = {}

    def fake_rerank(query, pointers, k):
        seen["slate"] = len(pointers)
        return list(reversed(pointers))[:k]  # deliberately invert the fused order

    results = retrieve(
        "descriptive question", k=3, conn=conn,
        embed_fn=lambda q: [0.0] * 384, rerank_fn=fake_rerank,
    )
    assert seen["slate"] == 10  # the whole fused slate, not just top-k
    assert [r["chunk_key"] for r in results] == ["ref9", "ref8", "ref7"]


def test_retrieve_skips_reranker_when_disabled(monkeypatch):
    monkeypatch.setenv("TORCHDOCS_RERANK", "0")
    refs = [_row(f"ref{i}", kind="api", dist=0.2) for i in range(4)]
    conn = FakeConn([refs, [], [], [], [], []])
    results = retrieve("q", k=2, conn=conn, embed_fn=lambda q: [0.0] * 384)
    # fused order preserved — no reranker touched it (none injected, env off)
    assert [r["chunk_key"] for r in results] == ["ref0", "ref1"]


def test_exact_symbol_pin_outranks_the_reranker():
    exact = _row("exact", kind="api", url="https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html")
    refs = [_row(f"ref{i}", kind="api", dist=0.2) for i in range(3)]
    # query pools: dense, keyword, then the symbol channel returns the exact page
    conn = FakeConn([refs, [], [], [], [], [], [exact]])

    def fake_rerank(query, pointers, k):
        return [p for p in pointers if p["chunk_key"] != "exact"][:k]  # reranker demotes it

    results = retrieve(
        "how do I use torch.optim.SGD?", k=3, conn=conn,
        embed_fn=lambda q: [0.0] * 384, rerank_fn=fake_rerank,
    )
    assert results[0]["chunk_key"] == "exact"  # the pin wins anyway
