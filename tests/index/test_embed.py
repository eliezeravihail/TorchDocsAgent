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


def test_symbol_from_url():
    from index.embed import symbol_from_url

    url = "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.sdpa.html"
    assert symbol_from_url(url) == "torch.nn.functional.sdpa"
    tut = "https://docs.pytorch.org/tutorials/beginner/intro.html"
    assert symbol_from_url(tut) == ""


def test_indexed_text_prepends_symbol_and_heading():
    from index.embed import indexed_text

    unit = {
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html",
        "heading_path": ["torch.optim.SGD", "Parameters"],
        "content": "lr (float) — learning rate.",
    }
    text = indexed_text(unit)
    assert text.startswith("torch.optim.SGD")
    assert "torch.optim.SGD > Parameters" in text
    assert "lr (float)" in text


def test_indexed_text_prepends_the_pages_gloss_when_present(monkeypatch):
    # Contextual Retrieval: the plain-language gloss is the semantic bridge
    # between a signature-shaped reference page and descriptive questions —
    # it must land in indexed_text (→ both the vector and the tsvector)
    from index.embed import indexed_text

    url = "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html"
    gloss = "Stochastic gradient descent optimizer with optional momentum."
    monkeypatch.setattr("index.embed.load_glosses", lambda: {url: gloss})
    unit = {"url": url, "heading_path": ["torch.optim.SGD"], "content": "lr (float)"}
    text = indexed_text(unit)
    assert gloss in text
    assert text.index("torch.optim.SGD") < text.index(gloss) < text.index("lr (float)")
    # a page with no gloss is unchanged
    assert gloss not in indexed_text({**unit, "url": url.replace("SGD", "Adam")})


def test_gloss_stamp_changes_the_recipe_with_gloss_content(monkeypatch, tmp_path):
    # any change to the committed gloss file must force the one-time full
    # re-embed by itself — a stale recipe would silently keep old vectors
    from index import embed

    monkeypatch.setattr(embed, "GLOSSES_PATH", tmp_path / "glosses.jsonl")
    assert embed._gloss_stamp() == "none"
    embed.GLOSSES_PATH.write_text('{"url": "u", "gloss": "g"}\n')
    first = embed._gloss_stamp()
    embed.GLOSSES_PATH.write_text('{"url": "u", "gloss": "different"}\n')
    assert first not in ("none", embed._gloss_stamp())


def test_iter_corpus_units_walks_snapshot(tmp_path):
    save_page("https://docs.pytorch.org/docs/stable/optim.html", "core", HTML, tmp_path)
    units = list(iter_corpus_units(tmp_path))
    assert units, "expected chunks from the snapshot page"
    assert all(u["url"].endswith("optim.html") for u in units)
    assert any("SGD" in " > ".join(u["heading_path"]) for u in units)


def test_chunk_key_part_zero_keeps_the_legacy_format():
    # backward compatibility is the whole point: introducing parts must not
    # change existing rows' keys, or the next build re-embeds the entire corpus
    from index.embed import chunk_key

    unit = {"url": "https://x/p.html", "anchor": "a", "heading_path": ["A", "B"]}
    assert chunk_key({**unit, "part": 0}) == chunk_key(unit)
    assert chunk_key({**unit, "part": 1}) != chunk_key(unit)
    assert chunk_key({**unit, "part": 1}) != chunk_key({**unit, "part": 2})
