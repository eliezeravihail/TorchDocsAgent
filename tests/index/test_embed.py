from index.embed import QUERY_PREFIX, batches, chunk_key, iter_corpus_units
from ingest.crawl import save_page

UNIT = {
    "url": "https://docs.pytorch.org/docs/stable/optim.html",
    "anchor": "torch-optim-sgd",
    "heading_path": ["torch.optim", "SGD"],
    "content_hash": "abc",
}

HTML = """<html><head><title>Optim</title></head><body><div role="main">
<h1>torch.optim</h1><p>Optimizers.</p><h2>SGD</h2><p>Stochastic gradient descent.</p>
</div></body></html>"""


def test_chunk_key_stable_and_distinct():
    assert chunk_key(UNIT) == chunk_key(dict(UNIT))
    other = {**UNIT, "anchor": "other"}
    assert chunk_key(other) != chunk_key(UNIT)


def test_batches_splits_exactly():
    items = list(range(10))
    got = list(batches(items, 4))
    assert got == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_query_prefix_is_bge_convention():
    assert QUERY_PREFIX.startswith("Represent this sentence")


def test_iter_corpus_units_walks_snapshot(tmp_path):
    save_page("https://docs.pytorch.org/docs/stable/optim.html", "core", HTML, tmp_path)
    units = list(iter_corpus_units(tmp_path))
    assert units, "expected chunks from the snapshot page"
    assert all(u["url"].endswith("optim.html") for u in units)
    assert any("SGD" in " > ".join(u["heading_path"]) for u in units)
