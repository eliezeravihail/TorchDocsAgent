"""Chunk snapshot pages by heading into OKF units.

A chunk = one coherent doc section: its heading path, its prose, and any code
blocks that live under it. API pages also carry their [source] GitHub link as
metadata. Units are emitted as OKF files (YAML frontmatter + markdown body) —
the human/agent-readable knowledge snapshot — before M2's embed step loads
them into Neon.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
GITHUB_SOURCE_RE = re.compile(r"https://github\.com/pytorch/[\w.-]+/blob/\S+")
# Sphinx headerlinks survive HTML→markdown as e.g. [¶](#sgd "Permalink...") —
# they carry the TRUE anchor, and must not leak into titles/heading paths
HEADERLINK_RE = re.compile(r"\[[^\]]*\]\(#([^)\s\"]+)[^)]*\)")


@dataclass
class Section:
    heading_path: list[str]
    title: str
    text: str
    anchor: str = ""
    source_link: str = field(default="")


def slugify(title: str) -> str:
    """Sphinx-style anchor slug: lowercase, alphanumerics, hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def clean_heading(raw_title: str) -> tuple[str, str]:
    """(clean title, anchor): take the real Sphinx anchor from the headerlink
    when present, strip the link markup from the title, unescape markdown."""
    match = HEADERLINK_RE.search(raw_title)
    title = HEADERLINK_RE.sub("", raw_title).replace("\\_", "_").strip()
    anchor = match.group(1) if match else slugify(title)
    return title, anchor


def split_by_heading(markdown: str) -> list[Section]:
    """Split a page into sections at every heading; preamble becomes section 0."""
    matches = list(HEADING_RE.finditer(markdown))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) breadcrumbs

    preamble = markdown[: matches[0].start()].strip() if matches else markdown.strip()
    if preamble:
        sections.append(Section(heading_path=[], title="", text=preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title, anchor = clean_heading(match.group(2).strip())
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        text = markdown[match.end() : end].strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        source = GITHUB_SOURCE_RE.search(text)
        sections.append(
            Section(
                heading_path=[t for _, t in stack],
                title=title,
                text=text,
                anchor=anchor,
                source_link=source.group(0) if source else "",
            )
        )
    return sections


def page_kind(url: str) -> str:
    if "/tutorials/" in url:
        return "tutorial"
    if "/generated/" in url or "/docs/" in url:
        return "api"
    return "guide"


def chunk_page(meta: dict, body: str) -> list[dict]:
    """One snapshot page → list of OKF-unit dicts (frontmatter fields + content)."""
    units = []
    for section in split_by_heading(body):
        if not section.text:
            continue
        units.append(
            {
                "url": meta["url"],
                "anchor": section.anchor,
                "page_title": meta.get("title", ""),
                "heading_path": section.heading_path,
                "library": meta.get("library", ""),
                "kind": page_kind(meta["url"]),
                "source_link": section.source_link,
                "content_hash": meta.get("content_hash", ""),
                "content": section.text,
            }
        )
    return units


def write_units(units: list[dict], out_dir: Path) -> list[Path]:
    """Write each unit as an OKF file: YAML frontmatter over the section body."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, unit in enumerate(units):
        frontmatter = {k: v for k, v in unit.items() if k != "content"}
        stem = slugify(f"{Path(unit['url']).stem}-{unit['anchor'] or i}")
        path = out_dir / f"{stem}.md"
        frontmatter_yaml = yaml.safe_dump(frontmatter, sort_keys=True)
        path.write_text(f"---\n{frontmatter_yaml}---\n\n{unit['content']}\n")
        paths.append(path)
    return paths
