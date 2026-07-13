---
title: "ingest/ — the crawl → snapshot → chunk pipeline"
kind: reference
package: ingest
---

# `ingest/` — the crawl → snapshot → chunk pipeline

Turns the live PyTorch documentation *site* into an on-disk knowledge snapshot: enumerate every page, fetch and clean it to markdown, and split it into heading-granular OKF units that the index step embeds.

## Why this package exists / its boundary

`ingest/` produces the **source of truth** for retrieval: the on-disk `_corpus/` snapshot (one markdown file per page) and, from it, the OKF chunk units. That is the whole job — **it never touches the database.** Embedding, tsvector computation, and the upsert into Neon all live downstream in `index/embed.py`, which *consumes* what this package writes.

The split is deliberate and load-bearing for the design's pointer-based storage (see `../design-content-and-agent-flow.md` §1.2): the DB stores only vectors, tsvectors, and pointers — **no page text**. At query time content is re-read ("hydrated") from the snapshot. So the snapshot isn't a scratch cache you can delete after indexing; it's a runtime dependency that ships with the deploy (PLAN.md M5). Keeping ingest DB-free means the crawl can run anywhere (a laptop, a CI runner) with no credentials, and the snapshot it emits is a plain, greppable, diff-able directory of markdown — the artifact a human can inspect to see exactly what the agent will ever be able to say.

## The pipeline

Three stages, each a separate module, each handing a concrete artifact to the next:

```
discover.py   →   crawl.py            →   chunk_docs.py        →  (index/embed.py)
enumerate URLs    fetch + clean + snap    split by heading         embed + upsert
                                                                   [downstream, not us]

{library:            _corpus/<path>.md         OKF units
 {url, url, …}}      ---                        ---
                     url, title, library,       url, anchor, page_title,
                     content_hash, crawled_at   heading_path, library, kind,
                     ---                         source_link, content_hash,
                     <clean markdown body>       part, synopsis
                                                 ---
                                                 <section body, code attached>
```

1. **discover** (`discover.py`) — for each seed in the tiered `SEEDS` map, fetch `objects.inv` (the Sphinx inventory: every documented symbol → exact page + anchor) and `sitemap.xml` (tutorials/guides no inventory covers). Returns `{library: set(page_url)}`. Only inventory entries ending in `.html`, and only sitemap URLs under the seed base, are kept.
2. **crawl** (`crawl.py`) — for each discovered URL: `fetch_html` (following the meta-refresh redirect stubs `docs/stable/` serves), strip nav/chrome to the main content container, convert HTML → markdown, hash the body, and write `_corpus/<url-path>.md` with YAML frontmatter. If the `content_hash` matches what's already on disk, the page is skipped — nothing downstream re-runs.
3. **chunk** (`chunk_docs.py`) — read each snapshot page, split it at every heading into `Section`s (code fences stay attached to their section), cut oversized sections at natural seams, and emit OKF unit dicts / files carrying the pointer metadata + a per-page synopsis.

## Design decisions & rationale

**Heading-granular chunks, never a token window.** `split_by_heading` cuts a page at its markdown headings; a chunk is one coherent section with its full `heading_path` breadcrumb. This is the kapa.ai lesson (design doc §Research grounding): a fixed-size token window slices a catalog list ("What LR schedulers exist?") or a worked example in half, and half a list retrieves as noise. Sections are the unit an author already made coherent. Oversized sections are *still* split — but by `split_oversized`, which greedily packs whole paragraphs and whole code fences up to `CHUNK_TARGET_CHARS` (2000, aligned with `index/embed.MAX_EMBED_CHARS`), and only ever cuts *inside* an atom that alone exceeds the budget, at line boundaries. Each resulting part inherits the section's `heading_path` and URL+anchor, so citations stay exact and every part is prefixed with the same symbol/heading synopsis.

**Code blocks are atoms.** `_atoms` walks the fenced-code regex so that the blank lines *inside* a `​```…```​` block never become split points. Half a code block is noise to embed; a fence is indivisible.

**The snapshot is the source of truth.** Covered above under boundary — the DB stores no page text, so the crawl output is not disposable. `save_page` writes human-readable markdown + frontmatter precisely so the snapshot doubles as an auditable knowledge artifact, not an opaque blob.

**`content_hash` idempotency / incrementality.** `save_page` sha256's the *rendered markdown body* and short-circuits if the on-disk file already carries that hash (`return False`, "unchanged"). This is the chat-langchain record-manager lesson (design doc §Research grounding): a weekly recrawl over thousands of mostly-unchanged pages must be cheap. Because chunk identity is `(url, anchor)` and the hash rides through into every unit's frontmatter, an unchanged page re-chunks and re-embeds to nothing downstream. Hashing the *body* (not the raw HTML) is the right choice — it ignores chrome/timestamp churn that doesn't change meaning.

**The tiered seed list.** `SEEDS` is the v1-core tier (core, tutorials, vision, audio) from design doc §1.1. The whole point, restated in the code comment: adding a doc set (ExecuTorch, torchao, …) is *one line here and nothing else*, because every PyTorch domain library is a Sphinx site with its own `objects.inv` that the same discover→crawl→chunk path already handles. The `library` field is stamped on every page and chunk so retrieval can filter/route per question.

**Why the docs SITE, not the source code.** The corpus is what `docs.pytorch.org` serves, never the PyTorch source tree. This supersedes an earlier five-source-modules plan (design doc scope-history note) and is externally validated: LangChain tried indexing their own source code for retrieval and dropped it — raw code chunks retrieved worse than prose docs. Source questions are handled by *referral* instead: `chunk_docs` captures each API page's `[source]` GitHub link (`GITHUB_SOURCE_RE`) as `source_link` metadata, and the agent's `ask_source` tool points at DeepWiki. The docs are the knowledge boundary; code is a link, not a claim.

**OKF units.** Chunks are emitted as Open Knowledge Format files — YAML frontmatter over a markdown body — not raw DB rows. This makes each chunk a human/agent-readable knowledge snapshot you can open and read, consistent with the repo-wide OKF convention (PLAN.md) for hand-authored/generated knowledge documents. (Note the deliberate limit: OKF is *not* used for the DB schema itself, which is pointer-based with typed columns — the units are the on-disk representation the embed step loads.)

## Tool & library choices

| Tool | Where | Why this one |
|---|---|---|
| **requests** | `discover.fetch` | Streaming GET with a hard `MAX_PAGE_BYTES` cap enforced mid-download, retry/backoff on 5xx/429 (permanent 4xx raises immediately), and a `torchdocs-agent` UA. Streaming lets a runaway body be abandoned early instead of after a full download. |
| **beautifulsoup4** | `crawl.extract_main_html`, `discover.redirect_target` | Select the Sphinx content container (`article.pytorch-article` → `div[role=main]` → … → `body`) and `decompose()` the chrome tags (`nav/header/footer/script/style/aside/form/iframe`); also to read the `<meta http-equiv=refresh>` / `<link rel=canonical>` redirect target. |
| **markdownify** | `crawl.to_markdown` | HTML fragment → markdown with ATX headings (`#`), `*` bullets. ATX headings are what `chunk_docs.HEADING_RE` splits on, so the two modules are coupled by this choice. |
| **PyYAML** | `crawl` + `chunk_docs` | `safe_dump`/`safe_load` the frontmatter on both snapshot pages and OKF units (`sort_keys=True` for stable diffs). |
| **Sphinx `objects.inv` parsing** | `discover.parse_objects_inv` | Hand-rolled on stdlib `zlib` + Sphinx's *own* inventory-line regex. The format is `"<name> <domain:role> <priority> <uri> <dispname>"` and **names may contain spaces** (`std:label "PyTorch Contribution Guide"`), so a naive split corrupts the URI — hence the exact upstream regex. Handles the `$` shorthand (uri ending in `$` expands to the entry name). |
| stdlib `xml.etree` | `discover.parse_sitemap` | Namespace-agnostic sitemap parsing (`_localname` strips ElementTree's `{ns}` prefix), following one level of `<sitemapindex>`, taking only the `<loc>` directly under each `<url>`/`<sitemap>` so nested `<image:loc>` is ignored. |

Two hard-won specifics worth calling out, both documented in code comments:

- **Redirect stubs.** `docs.pytorch.org/docs/stable/<…>` serves a "Redirecting…" page whose real content is behind a `<meta refresh>` — which `requests` does *not* follow. A naive fetch captured empty stubs for 3,435 of 4,517 pages (the entire core API reference). `fetch_html` detects the refresh meta and follows it, hop-bounded and loop-protected, while the caller keeps the *original* URL as the citation key so stable URLs stay stable.
- **The size cap is generous on purpose.** `MAX_PAGE_BYTES` is 20 MB, not 5 — a 5 MB cap silently dropped legitimate image-heavy tutorials (dcgan_faces, hybrid_demucs) from the index on the 2026-07-08 build.

## File by file

All four modules are **implemented** (with a full test suite under `tests/ingest/` — `test_discover.py`, `test_crawl.py`, `test_chunk_docs.py`). Note PLAN.md's M2.1 checkboxes for these files were still unticked at the time of writing; the code is present and substantive, so treat the boxes as lagging the work, not the reverse.

- **`__init__.py`** — empty (0 bytes). Marks `ingest` as a package; there is no package-level API surface, callers import the modules directly (e.g. `from ingest.discover import fetch_html`).

- **`discover.py`** — page enumeration. Pure parsers (`parse_objects_inv`, `parse_sitemap`, `is_sitemap_index`, `redirect_target`) take bytes/text and return entries, so they test offline; only `fetch`/`fetch_html`/`discover` touch the network. `discover()` walks `SEEDS`, tries inventory then sitemap per seed, and *tolerates only network errors* — a parse error (PyTorch changing the inventory format) propagates and fails the run loudly rather than silently shrinking the index. Also home to the resilient `fetch` (retry/backoff/size-cap) and the meta-refresh-following `fetch_html`.

- **`crawl.py`** — fetch → clean → snapshot. `extract_main_html` selects the content container and strips chrome; `to_markdown` converts; `page_path` maps a URL to a stable on-disk path (host dropped, path preserved, `.html`→`.md`); `save_page` hashes the body and writes-or-skips; `crawl` drives the whole set with a politeness delay (`TORCHDOCS_CRAWL_DELAY`, default 0.2s) and per-page exception isolation (one bad page must not kill the crawl). Returns per-library change counts.

- **`chunk_docs.py`** — snapshot page → OKF units. `split_by_heading` builds `Section`s with a breadcrumb stack; `clean_heading` recovers the *true* Sphinx anchor from the surviving headerlink (`[¶](#sgd …)`) rather than re-slugifying; `split_oversized`/`_atoms`/`_hard_split` handle over-budget sections without cutting code fences; `page_kind` classifies url → `api`/`tutorial`/`guide`; `page_synopsis` deterministically extracts the page's first real description sentence (the docstring summary buried after Sphinx's `:` definition-line marker) — API pages only, since terse reference pages otherwise embed far from descriptive questions. `chunk_page` assembles the unit dicts; `write_units` serializes them, always suffixing the filename with the enumerate index so two same-named sections (e.g. two "Parameters" headings) can't clobber each other on disk.

- **`watch.py`** — **not present / not yet built.** The design doc (§1.3) and PLAN.md M5 specify a release watcher that polls the `pytorch/pytorch` GitHub Releases API and kicks an immediate recrawl on a new stable tag. It does not exist in `ingest/` yet; the scheduled/triggered recrawl orchestration is M5 work.

## Related docs

- `../design-content-and-agent-flow.md` — the design rationale this package implements, especially §1 (content extraction), §1.1 (the tiered corpus/seed list), and §1.2–1.3 (pipeline properties and recrawl cadence).
- `../index/README.md` — the sibling `index/` package (`embed.py`, `hydrate.py`) that consumes the snapshot: embeds the OKF units into Neon and hydrates content back at query time.
