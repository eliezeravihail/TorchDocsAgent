from index.hydrate import hydrate_page, hydrate_section
from ingest.crawl import save_page

URL = "https://docs.pytorch.org/docs/stable/optim.html"
HTML = """<html><head><title>Optim</title></head><body><div role="main">
<h1>torch.optim</h1><p>Optimizers overview.</p>
<h2>SGD</h2><p>Stochastic gradient descent details.</p>
</div></body></html>"""


def _snapshot(tmp_path):
    save_page(URL, "core", HTML, tmp_path)


def test_hydrate_section_returns_matching_content(tmp_path):
    _snapshot(tmp_path)
    pointer = {"url": URL, "heading_path": "torch.optim > SGD"}
    result = hydrate_section(pointer, tmp_path)
    assert result is not None
    assert "Stochastic gradient descent" in result["content"]


def test_hydrate_section_missing_page_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("index.hydrate._LIVE", False)  # no live fallback in this test
    assert hydrate_section({"url": URL, "heading_path": "x"}, tmp_path) is None


def test_hydrate_section_heading_gone_returns_none_not_preamble(tmp_path):
    # regression: when the pointer's heading no longer matches, we must NOT
    # substitute the page preamble under the section's citation — return None
    _snapshot(tmp_path)
    pointer = {"url": URL, "heading_path": "torch.optim > RMSprop (renamed)"}
    assert hydrate_section(pointer, tmp_path) is None


def test_hydrate_page_returns_full_content(tmp_path):
    _snapshot(tmp_path)
    result = hydrate_page(URL, tmp_path)
    assert "Optimizers overview" in result["content"]
    assert "Stochastic gradient descent" in result["content"]


def test_hydrate_page_oversized_returns_outline(tmp_path, monkeypatch):
    _snapshot(tmp_path)
    monkeypatch.setattr("index.hydrate.PAGE_CHAR_LIMIT", 10)
    result = hydrate_page(URL, tmp_path)
    assert "content" not in result
    assert any("SGD" in item for item in result["outline"])


def test_hydrate_section_returns_the_pointed_part(tmp_path, monkeypatch):
    # a size-split section: the pointer's `part` selects WHICH slice comes back
    import ingest.chunk_docs as cd

    monkeypatch.setattr(cd, "CHUNK_TARGET_CHARS", 120)
    # each paragraph fits the limit alone; together they overflow -> 2 parts
    html = (
        "<html><head><title>Big</title></head><body><div role=\"main\">"
        "<h1>Guide</h1><p>" + "alpha " * 15 + "</p><p>" + "omega " * 15 + "</p>"
        "</div></body></html>"
    )
    url = "https://docs.pytorch.org/docs/stable/big.html"
    save_page(url, "core", html, tmp_path)

    first = hydrate_section({"url": url, "heading_path": "Guide", "part": 0}, tmp_path)
    second = hydrate_section({"url": url, "heading_path": "Guide", "part": 1}, tmp_path)
    assert first and "alpha" in first["content"] and "omega" not in first["content"]
    assert second and "omega" in second["content"] and "alpha" not in second["content"]
    # a pointer to a part that no longer exists drops cleanly
    assert hydrate_section({"url": url, "heading_path": "Guide", "part": 9}, tmp_path) is None
