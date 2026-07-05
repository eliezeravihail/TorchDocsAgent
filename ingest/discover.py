"""Enumerate every page of the docs corpus.

Two discovery sources, per the design doc:
- Sphinx ``objects.inv`` per doc set — the authoritative symbol → (page, anchor)
  map, covering the whole API reference of core/vision/audio/etc.
- ``sitemap.xml`` — tutorials and guide pages that no inventory covers.

Parsing is pure (bytes in, entries out) so it tests offline; only fetch()
touches the network.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass

# Sphinx's own inventory-line pattern (sphinx/util/inventory.py)
_INV_LINE_RE = re.compile(r"(.+?)\s+(\S+)\s+(-?\d+)\s+?(\S*)\s+(.*)")

# Tiered seed list from docs/design-content-and-agent-flow.md §1.1 (v1 core).
# Adding a doc set (ExecuTorch, torchao, ...) is one line here — nothing else.
SEEDS: dict[str, str] = {
    "core": "https://docs.pytorch.org/docs/stable/",
    "tutorials": "https://docs.pytorch.org/tutorials/",
    "vision": "https://docs.pytorch.org/vision/stable/",
    "audio": "https://docs.pytorch.org/audio/stable/",
}


@dataclass(frozen=True)
class InvEntry:
    """One documented object from a Sphinx inventory."""

    name: str  # e.g. "torch.nn.Linear"
    role: str  # e.g. "py:class"
    page_url: str  # absolute page URL, no fragment
    anchor: str  # fragment, "" if none


def parse_objects_inv(data: bytes, base_url: str) -> list[InvEntry]:
    """Parse a Sphinx v2 inventory into entries with absolute URLs."""
    header, _, rest = data.partition(b"\n")
    if not header.startswith(b"# Sphinx inventory version 2"):
        raise ValueError(f"unsupported inventory header: {header[:50]!r}")
    for _ in range(3):  # Project / Version / compression-note comment lines
        _, _, rest = rest.partition(b"\n")

    base = base_url.rstrip("/") + "/"
    entries: list[InvEntry] = []
    for line in zlib.decompress(rest).decode("utf-8").splitlines():
        # format: "<name> <domain:role> <priority> <uri> <dispname>".
        # NAME MAY CONTAIN SPACES (e.g. std:label "PyTorch Contribution Guide"),
        # so a naive split corrupts the uri — this is Sphinx's own regex.
        match = _INV_LINE_RE.match(line.rstrip())
        if match is None:
            continue
        name, role, _priority, uri, _dispname = match.groups()
        if uri.endswith("$"):  # Sphinx shorthand: '$' expands to the entry name
            uri = uri[:-1] + name
        page, _, anchor = uri.partition("#")
        entries.append(InvEntry(name=name, role=role, page_url=base + page, anchor=anchor))
    return entries


def parse_sitemap(xml_text: str) -> list[str]:
    """Extract <loc> URLs from a sitemap (namespace-agnostic)."""
    root = ET.fromstring(xml_text)
    return [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text]


def fetch(url: str, timeout: float = 30.0) -> bytes:
    import requests

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "torchdocs-agent"})
    response.raise_for_status()
    return response.content


def discover(seeds: dict[str, str] | None = None) -> dict[str, set[str]]:
    """Return {library: set of page URLs} for the whole corpus.

    Per seed: try objects.inv first (API reference), then sitemap.xml
    (tutorials/guides). A seed that yields neither is reported empty rather
    than failing the whole run.
    """
    pages: dict[str, set[str]] = {}
    for library, base in (seeds or SEEDS).items():
        found: set[str] = set()
        try:
            entries = parse_objects_inv(fetch(base + "objects.inv"), base)
            # defense in depth: real doc pages end in .html; anything else is
            # a malformed entry and would just 404 in the crawl
            found.update(e.page_url for e in entries if e.page_url.endswith(".html"))
        except Exception as exc:  # noqa: BLE001 — a missing inventory must not kill the run
            print(f"[discover] {library}: no objects.inv ({exc})")
        try:
            found.update(
                url
                for url in parse_sitemap(fetch(base + "sitemap.xml").decode("utf-8"))
                if url.startswith(base)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[discover] {library}: no sitemap ({exc})")
        pages[library] = found
        print(f"[discover] {library}: {len(found)} pages")
    return pages
