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


def _localname(tag: str) -> str:
    """Strip the ``{namespace}`` prefix ElementTree prepends to tags."""
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(xml_text: str) -> list[str]:
    """Extract page/sub-sitemap <loc> URLs from a sitemap (namespace-agnostic).

    Handles both a <urlset> (page URLs) and a <sitemapindex> (child sitemap
    URLs). Only the <loc> directly under each <url>/<sitemap> is taken, so
    nested <image:loc> entries are ignored rather than mistaken for pages.
    """
    root = ET.fromstring(xml_text)
    urls: list[str] = []
    for entry in root:
        if _localname(entry.tag) not in ("url", "sitemap"):
            continue
        loc = next((c for c in entry if _localname(c.tag) == "loc" and c.text), None)
        if loc is not None:
            urls.append(loc.text.strip())
    return urls


def is_sitemap_index(xml_text: str) -> bool:
    return _localname(ET.fromstring(xml_text).tag) == "sitemapindex"


FETCH_RETRIES = 3
# Guards against runaway bodies (a tarball link, a broken page) ballooning
# memory across a thousands-page crawl. Generous on purpose: legitimate
# tutorials with inline images (dcgan_faces, hybrid_demucs) run 5-10MB, and a
# 5MB cap silently dropped them from the index on the 2026-07-08 build.
MAX_PAGE_BYTES = 20 * 1024 * 1024


def fetch(url: str, timeout: float = 30.0, retries: int = FETCH_RETRIES) -> bytes:
    """GET with retry/backoff on transient failures and a hard size cap.

    Retries cover network errors, 5xx, and 429; other 4xx (404, 403) are
    permanent for a crawl and raise immediately. The size cap is enforced
    while streaming, so a huge body is abandoned early, not after download.
    """
    import time

    import requests

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": "torchdocs-agent"}, stream=True
            )
            response.raise_for_status()
            body = b""
            for chunk in response.iter_content(65536):
                body += chunk
                if len(body) > MAX_PAGE_BYTES:
                    raise ValueError(f"{url}: page exceeds {MAX_PAGE_BYTES} bytes; skipping")
            return body
        except requests.RequestException as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise  # permanent for a crawl (404 gone, 403 blocked) — retrying won't help
            time.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]  # loop ran ≥1 time, so last_exc is set


def redirect_target(html: str, base_url: str) -> str | None:
    """The URL a client-side redirect stub points at, else None.

    docs.pytorch.org/docs/stable/<...> serves a "Redirecting…" page whose only
    real content is `<meta http-equiv="refresh" content="0; url=<versioned>">`.
    requests follows HTTP 3xx but NOT this, so a plain fetch captured an empty
    stub for the entire core API reference (measured: 3,435/4,517 crawled pages
    had title "Redirecting…", every one a docs/stable API page). A real content
    page never carries a refresh meta, so finding one unambiguously means "this
    is a redirect — follow it". Falls back to <link rel="canonical"> when it
    points elsewhere.
    """
    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"http-equiv": re.compile(r"^refresh$", re.I)})
    if meta and meta.get("content"):
        m = re.search(r"url\s*=\s*(.+)$", meta["content"], re.I)
        if m:
            return urljoin(base_url, m.group(1).strip().strip("'\""))
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        target = urljoin(base_url, canonical["href"].strip())
        if target.rstrip("/") != base_url.rstrip("/"):
            return target
    return None


def fetch_html(url: str, *, max_hops: int = 3, **kwargs) -> str:
    """fetch() + follow client-side (meta-refresh) redirects to the real page.

    Returns the final page's HTML as text. Loop-protected and hop-bounded so a
    misconfigured redirect chain can't spin forever. The caller keeps its
    ORIGINAL url as the pointer/citation key — only the *content* comes from the
    redirect target — so stable URLs stay stable in the index and in answers.
    """
    seen: set[str] = set()
    html = ""
    for _ in range(max_hops + 1):
        html = fetch(url, **kwargs).decode("utf-8", errors="replace")
        seen.add(url)
        target = redirect_target(html, url)
        if not target or target in seen:
            return html
        print(f"[fetch] following redirect {url} → {target}", flush=True)
        url = target
    return html  # hop budget exhausted → return the last page fetched


def _sitemap_pages(base: str, xml_text: str) -> set[str]:
    """Page URLs from a sitemap, following one level of <sitemapindex>."""
    import requests

    if not is_sitemap_index(xml_text):
        return set(parse_sitemap(xml_text))
    pages: set[str] = set()
    for sub in parse_sitemap(xml_text):  # child sitemap .xml URLs
        try:
            pages.update(parse_sitemap(fetch(sub).decode("utf-8")))
        except requests.RequestException as exc:
            print(f"[discover] sub-sitemap {sub} unreachable ({exc})")
    return pages


def discover(seeds: dict[str, str] | None = None) -> dict[str, set[str]]:
    """Return {library: set of page URLs} for the whole corpus.

    Per seed: try objects.inv first (API reference), then sitemap.xml
    (tutorials/guides). Only NETWORK errors are tolerated (a genuinely missing
    inventory/sitemap) — a parse error (PyTorch changing the inventory format)
    propagates and fails the run loudly rather than silently shrinking the index.
    """
    import requests

    pages: dict[str, set[str]] = {}
    for library, base in (seeds or SEEDS).items():
        found: set[str] = set()
        try:
            inv = fetch(base + "objects.inv")
        except requests.RequestException as exc:
            print(f"[discover] {library}: no objects.inv ({exc})")
        else:
            # defense in depth: real doc pages end in .html; anything else is
            # a malformed entry and would just 404 in the crawl
            entries = parse_objects_inv(inv, base)
            found.update(e.page_url for e in entries if e.page_url.endswith(".html"))
        try:
            sitemap = fetch(base + "sitemap.xml").decode("utf-8")
        except requests.RequestException as exc:
            print(f"[discover] {library}: no sitemap ({exc})")
        else:
            found.update(u for u in _sitemap_pages(base, sitemap) if u.startswith(base))
        pages[library] = found
        print(f"[discover] {library}: {len(found)} pages")
    return pages
