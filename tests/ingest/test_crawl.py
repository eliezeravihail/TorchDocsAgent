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


def test_crawl_is_polite_between_requests(monkeypatch, tmp_path):
    import time

    import ingest.crawl as crawl_mod

    monkeypatch.setenv("TORCHDOCS_CRAWL_DELAY", "0.5")
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        "ingest.discover.fetch", lambda url: b"<html><body><p>hi</p></body></html>"
    )
    crawl_mod.crawl({"core": {"https://x/a.html", "https://x/b.html"}}, tmp_path)
    assert sleeps == [0.5, 0.5]  # one pause after every request, failures included
