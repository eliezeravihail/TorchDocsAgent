from agent.tools import ask_source, read_page


def test_ask_source_returns_referrals_without_network():
    result = ask_source("how is conv2d implemented internally?")
    urls = [r.url for r in result["referrals"]]
    assert any("deepwiki.com/pytorch/pytorch" in u for u in urls)
    assert any("github.com/search" in u and "pytorch" in u for u in urls)
    # never claims to know the code
    assert "referrals" in result and "note" in result


def test_ask_source_keeps_discriminating_terms_past_the_first_six_words():
    # regression: the old code kept only the first 6 words, dropping the actual
    # subject of longer questions; stopwords go, meaningful terms stay
    from agent.tools import _search_terms

    terms = _search_terms(
        "how is the backward pass of grouped convolution conv2d implemented in the source"
    )
    assert "grouped" in terms and "convolution" in terms and "conv2d" in terms
    assert "backward" in terms  # would have been dropped by the old [:6] slice
    assert "how" not in terms and "the" not in terms  # stopwords removed


def test_search_docs_shape(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr("agent.translate.translate_to_english", lambda q, **k: q)
    monkeypatch.setattr(
        "index.retrieve.retrieve",
        lambda q, k=8, library=None: [{"url": "u", "anchor": "a", "heading_path": "H"}],
    )
    monkeypatch.setattr(
        "index.hydrate.hydrate_section",
        lambda p: {**p, "content": "SGD implements gradient descent"},
    )
    result = tools.search_docs("how do I use SGD")
    assert result["sections"][0]["content"].startswith("SGD")
    assert result["titles"] == ["H"]


def test_read_page_missing(monkeypatch):
    monkeypatch.setattr("index.hydrate.hydrate_page", lambda url: None)
    assert "error" in read_page("https://x")
