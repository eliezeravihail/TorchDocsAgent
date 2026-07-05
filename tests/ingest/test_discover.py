import zlib

import pytest

from ingest.discover import InvEntry, parse_objects_inv, parse_sitemap

BASE = "https://docs.pytorch.org/docs/stable/"


def make_inv(lines: list[str]) -> bytes:
    header = (
        b"# Sphinx inventory version 2\n"
        b"# Project: PyTorch\n"
        b"# Version: 2.12\n"
        b"# The remainder of this file is compressed using zlib.\n"
    )
    return header + zlib.compress("\n".join(lines).encode())


def test_parse_objects_inv_expands_dollar_and_anchor():
    inv = make_inv(
        [
            "torch.nn.Linear py:class 1 generated/torch.nn.Linear.html#$ -",
            "torch.optim py:module 0 optim.html#module-torch.optim Optim docs",
        ]
    )
    entries = parse_objects_inv(inv, BASE)
    assert entries[0] == InvEntry(
        name="torch.nn.Linear",
        role="py:class",
        page_url=f"{BASE}generated/torch.nn.Linear.html",
        anchor="torch.nn.Linear",
    )
    assert entries[1].anchor == "module-torch.optim"
    assert entries[1].page_url == f"{BASE}optim.html"


def test_parse_objects_inv_name_with_spaces():
    # std:label / std:doc names may contain spaces — the naive split bug
    # turned words of the name into fake page URLs (each 404ing in the crawl)
    inv = make_inv(
        [
            "PyTorch Contribution Guide std:doc -1 community/contribution_guide.html "
            "PyTorch Contribution Guide",
            "torch.nn.Linear py:class 1 generated/torch.nn.Linear.html#$ -",
        ]
    )
    entries = parse_objects_inv(inv, BASE)
    assert entries[0].name == "PyTorch Contribution Guide"
    assert entries[0].page_url == f"{BASE}community/contribution_guide.html"
    assert entries[1].page_url == f"{BASE}generated/torch.nn.Linear.html"


def test_parse_objects_inv_rejects_wrong_version():
    with pytest.raises(ValueError, match="unsupported"):
        parse_objects_inv(b"# Sphinx inventory version 1\nrest", BASE)


def test_parse_sitemap():
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://docs.pytorch.org/tutorials/beginner/basics/intro.html</loc></url>
      <url><loc>https://docs.pytorch.org/tutorials/advanced/cpp.html</loc></url>
    </urlset>"""
    urls = parse_sitemap(xml)
    assert len(urls) == 2
    assert urls[0].endswith("intro.html")
