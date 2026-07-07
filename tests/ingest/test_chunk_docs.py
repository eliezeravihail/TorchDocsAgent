from ingest.chunk_docs import chunk_page, page_kind, split_by_heading, write_units

PAGE_MD = """Intro paragraph before any heading.

# torch.optim.SGD

Implements stochastic gradient descent.
[source](https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py#L26)

## Parameters

* lr (float) — learning rate

```python
optimizer = torch.optim.SGD(params, lr=0.01)
```

## Notes

Momentum is optional.

# See also

Other optimizers.
"""

META = {
    "url": "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html",
    "title": "torch.optim.SGD — PyTorch docs",
    "library": "core",
    "content_hash": "abc123",
}


def test_clean_heading_extracts_sphinx_anchor():
    from ingest.chunk_docs import clean_heading

    title, anchor = clean_heading('SGD[\u00b6](#torch-optim-sgd "Permalink to this heading")')
    assert title == "SGD"
    assert anchor == "torch-optim-sgd"
    title, anchor = clean_heading("Plain Heading")
    assert (title, anchor) == ("Plain Heading", "plain-heading")
    title, _ = clean_heading('get\\_tokenizer[\u00b6](#get-tokenizer "p")')
    assert title == "get_tokenizer"


def test_split_by_heading_paths_and_anchors():
    sections = split_by_heading(PAGE_MD)
    assert sections[0].heading_path == []  # preamble
    by_title = {s.title: s for s in sections if s.title}
    assert by_title["Parameters"].heading_path == ["torch.optim.SGD", "Parameters"]
    assert by_title["Notes"].heading_path == ["torch.optim.SGD", "Notes"]
    assert by_title["See also"].heading_path == ["See also"]  # sibling h1 resets stack
    assert by_title["Parameters"].anchor == "parameters"
    assert "optimizer = torch.optim.SGD" in by_title["Parameters"].text


def test_source_link_captured():
    sections = split_by_heading(PAGE_MD)
    sgd = next(s for s in sections if s.title == "torch.optim.SGD")
    assert sgd.source_link.startswith("https://github.com/pytorch/pytorch/blob/")


def test_chunk_page_units_and_kind():
    units = chunk_page(META, PAGE_MD)
    assert all(u["url"] == META["url"] for u in units)
    assert all(u["kind"] == "api" for u in units)
    assert page_kind("https://docs.pytorch.org/tutorials/beginner/intro.html") == "tutorial"


def test_write_units_valid_okf(tmp_path):
    import yaml

    paths = write_units(chunk_page(META, PAGE_MD), tmp_path)
    assert paths
    _, frontmatter, body = paths[0].read_text(encoding="utf-8").split("---\n", 2)
    meta = yaml.safe_load(frontmatter)
    assert {"url", "anchor", "heading_path", "library", "kind"} <= set(meta)
    assert body.strip()


def test_write_units_same_anchor_sections_dont_collide(tmp_path):
    # regression: two sections on one page can slugify to the same anchor
    # (e.g. two "Parameters" headings); each must get its own file, not clobber
    units = [
        {"url": META["url"], "anchor": "parameters", "heading_path": ["A", "Parameters"],
         "library": "core", "kind": "api", "content_hash": "h", "content": "first section"},
        {"url": META["url"], "anchor": "parameters", "heading_path": ["B", "Parameters"],
         "library": "core", "kind": "api", "content_hash": "h", "content": "second section"},
    ]
    paths = write_units(units, tmp_path)
    assert len(paths) == 2
    assert len(set(paths)) == 2  # distinct files, nothing overwritten
    bodies = {p.read_text(encoding="utf-8").split("---\n", 2)[2].strip() for p in paths}
    assert bodies == {"first section", "second section"}
