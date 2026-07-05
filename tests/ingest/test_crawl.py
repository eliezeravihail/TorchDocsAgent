from ingest.crawl import extract_main_html, load_page, page_path, save_page, to_markdown

HTML = """<html><head><title>torch.optim.SGD — PyTorch docs</title></head><body>
<nav><a href="/">home</a> lots of nav noise</nav>
<div role="main">
  <h1>torch.optim.SGD</h1>
  <p>Implements stochastic gradient descent.</p>
  <pre><code>optimizer = torch.optim.SGD(params, lr=0.01)</code></pre>
</div>
<footer>© PyTorch</footer>
</body></html>"""

URL = "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html"


def test_extract_main_strips_chrome():
    title, main = extract_main_html(HTML)
    assert "SGD" in title
    assert "nav noise" not in main
    assert "©" not in main
    assert "stochastic gradient descent" in main


def test_to_markdown_keeps_heading_and_code():
    _, main = extract_main_html(HTML)
    md = to_markdown(main)
    assert "# torch.optim.SGD" in md
    assert "optimizer = torch.optim.SGD(params, lr=0.01)" in md


def test_page_path_mirrors_url(tmp_path):
    assert page_path(URL, tmp_path) == tmp_path / "docs/stable/generated/torch.optim.SGD.md"


def test_save_page_roundtrip_and_hash_skip(tmp_path):
    assert save_page(URL, "core", HTML, tmp_path) is True
    meta, body = load_page(page_path(URL, tmp_path))
    assert meta["url"] == URL
    assert meta["library"] == "core"
    assert len(meta["content_hash"]) == 64
    assert "stochastic gradient descent" in body
    # identical content → skipped, not rewritten
    assert save_page(URL, "core", HTML, tmp_path) is False
    # changed content → written again
    assert save_page(URL, "core", HTML.replace("0.01", "0.02"), tmp_path) is True
