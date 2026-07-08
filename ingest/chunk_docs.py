"""Chunk snapshot pages by heading into OKF units.

A chunk = one coherent doc section: its heading path, its prose, and any code
blocks that live under it. API pages also carry their [source] GitHub link as
metadata. Units are emitted as OKF files (YAML frontmatter + markdown body) —
the human/agent-readable knowledge snapshot — which the embed step then loads
into Neon.
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

# A section bigger than this is cut into parts at natural seams. Aligned with
# index/embed.MAX_EMBED_CHARS: anything past it was invisible to the dense
# vector anyway (embedded blind, silently). Each part inherits the section's
# heading_path, so indexed_text() prepends the same symbol+heading "synopsis"
# to every part — the serialized-story recap — and all parts share the
# section's URL+anchor, so citations stay exact.
CHUNK_TARGET_CHARS = 2000
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


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


def _atoms(text: str) -> list[str]:
    """Paragraphs of the text, with fenced code blocks kept whole.

    A naive split on blank lines would cut inside ``` fences (code often has
    blank lines); half a code block is noise to embed. Fences are atoms.
    """
    atoms: list[str] = []
    last = 0
    for match in _FENCE_RE.finditer(text):
        atoms += [p for p in text[last : match.start()].split("\n\n") if p.strip()]
        atoms.append(match.group(0))
        last = match.end()
    atoms += [p for p in text[last:].split("\n\n") if p.strip()]
    return atoms


def _hard_split(atom: str, limit: int) -> list[str]:
    """Last resort for one atom bigger than the whole budget: cut at line ends."""
    parts: list[str] = []
    current = ""
    for line in atom.splitlines(keepends=True):
        while len(line) > limit:  # a single enormous line — slice it
            parts.append(line[:limit])
            line = line[limit:]
        if current and len(current) + len(line) > limit:
            parts.append(current)
            current = line
        else:
            current += line
    if current.strip():
        parts.append(current)
    return [p.strip("\n") for p in parts if p.strip()]


def split_oversized(text: str, limit: int | None = None) -> list[str]:
    """Cut oversized section text into parts at natural seams.

    Greedy paragraph packing: whole paragraphs (and whole code fences) are
    accumulated until the next one would overflow the limit; only an atom
    that alone exceeds the limit is cut inside (at line boundaries).
    """
    if limit is None:
        limit = CHUNK_TARGET_CHARS
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for atom in _atoms(text):
        pieces.extend(_hard_split(atom, limit) if len(atom) > limit else [atom])
    parts: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > limit:
            parts.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        parts.append(current)
    return parts


def page_kind(url: str) -> str:
    if "/tutorials/" in url:
        return "tutorial"
    if "/generated/" in url or "/docs/" in url:
        return "api"
    return "guide"


# a signature-shaped paragraph: `class torch.nn.Linear(...)`, `torch.add(...)`
_SIGNATURE_START_RE = re.compile(r"^(class\s+|@)?[\w.]+\(")
_SENTENCE_RE = re.compile(r"(.+?[.!?])(?:\s|$)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) → text
SYNOPSIS_MAX_CHARS = 240


def _plain(text: str) -> str:
    """Markdown → comparable prose: links keep their text, emphasis dropped."""
    return " ".join(_MD_LINK_RE.sub(r"\1", text).replace("*", "").replace("\\_", "_").split())


def page_synopsis(body: str, limit: int = SYNOPSIS_MAX_CHARS) -> str:
    """The page's own first prose sentence — the docstring summary a human wrote.

    Reference pages are signature/parameter-shaped, so their embedding sits far
    from descriptive questions even though the description sentence ("This
    criterion computes the cross entropy loss between input logits and target")
    usually exists — buried. Extracting it deterministically (no LLM, no quota,
    no hallucination) lets indexed_text() lead every chunk of the page with it.

    Shape gotcha this handles: markdownify renders Sphinx's signature <dl> as
    ONE paragraph — `*class* torch.nn.X(...)[[source]](github)` and then the
    description on a `:   ...` definition line. The description after the `:`
    marker is the synopsis; naively taking the paragraph head yields the
    signature + a github URL, which merely duplicates the chunk's own opening.
    """
    text = _FENCE_RE.sub("", body)
    for para in text.split("\n\n"):
        # a definition-list description line (`:   prose`) beats the paragraph
        # head — that's where Sphinx puts the docstring summary
        desc_lines = [ln.lstrip()[1:] for ln in para.splitlines() if ln.lstrip().startswith(": ")]
        flat = _plain(" ".join(desc_lines) if desc_lines else para)
        if not flat or flat.startswith("#"):
            continue
        if _SIGNATURE_START_RE.match(flat):
            continue
        if len(flat.split()) < 5:  # crumbs like "Parameters" or a bare type
            continue
        match = _SENTENCE_RE.match(flat)
        return (match.group(1) if match else flat)[:limit]
    return ""


def chunk_page(meta: dict, body: str) -> list[dict]:
    """One snapshot page → list of OKF-unit dicts (frontmatter fields + content).

    Oversized sections are cut into parts (split_oversized); every part keeps
    the section's heading_path (its embedded "synopsis") and URL+anchor (its
    citation), and carries a `part` ordinal that makes its identity unique.
    """
    units = []
    kind = page_kind(meta["url"])
    # api pages only: tutorials/guides already retrieve well, and their first
    # sentence is prose anyway — the synopsis is the terse pages' rescue line
    synopsis = page_synopsis(body) if kind == "api" else ""
    for section in split_by_heading(body):
        if not section.text:
            continue
        for part, content in enumerate(split_oversized(section.text)):
            units.append(
                {
                    "url": meta["url"],
                    "anchor": section.anchor,
                    "page_title": meta.get("title", ""),
                    "heading_path": section.heading_path,
                    "library": meta.get("library", ""),
                    "kind": kind,
                    "source_link": section.source_link,
                    "content_hash": meta.get("content_hash", ""),
                    "part": part,
                    "synopsis": synopsis,
                    "content": content,
                }
            )
    return units


def write_units(units: list[dict], out_dir: Path) -> list[Path]:
    """Write each unit as an OKF file: YAML frontmatter over the section body."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, unit in enumerate(units):
        frontmatter = {k: v for k, v in unit.items() if k != "content"}
        # always suffix with the enumerate index: two sections on one page can
        # slugify to the same anchor (e.g. two "Parameters" headings) and would
        # otherwise clobber each other on disk, silently losing a section
        stem = slugify(f"{Path(unit['url']).stem}-{unit['anchor'] or 'sec'}-{i}")
        path = out_dir / f"{stem}.md"
        frontmatter_yaml = yaml.safe_dump(frontmatter, sort_keys=True)
        path.write_text(f"---\n{frontmatter_yaml}---\n\n{unit['content']}\n", encoding="utf-8")
        paths.append(path)
    return paths
