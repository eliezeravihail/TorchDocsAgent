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
