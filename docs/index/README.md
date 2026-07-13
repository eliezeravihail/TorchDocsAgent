---
title: "The index/ package — retrieval + storage layer"
kind: reference
package: index
---

# The `index/` package

The Neon/pgvector storage and retrieval layer: it owns the `chunks` table, embeds
corpus chunks with a local bge-small model, serves hybrid (dense + keyword) search
as ranked pointers, hydrates section/page content, and self-heals drifted chunks
against the live docs after an answer goes out.

## Why this package exists / its boundary

Three packages split the RAG core along a clean seam, and `index/` is the middle one:

| Package | Owns | Hands off |
|---|---|---|
| `ingest/` | Crawls the docs site, builds the on-disk `_corpus/` snapshot, chunks pages by heading | Chunk *units* (`{url, anchor, heading_path, content, content_hash, kind, library, …}`) |
| **`index/`** | **Neon schema, embeddings, the `chunks` table, hybrid retrieval, hydration, freshness** | **Ranked *pointers* with content already attached** |
| `agent/` | The tool-calling loop, grounding, generation, guards | Answers with citations |

`ingest/` decides *what the corpus is*; `index/` decides *how it is stored and found*;
`agent/` decides *what to say*. The boundary is deliberate: `index/` never crawls
(that is `ingest/`'s job — see `_live_units` and `hydrate` importing `ingest.crawl`
rather than re-implementing it) and never reasons about answers (it returns pointers,
not prose). The one place this package reaches back into `ingest/` is to re-use the
*exact* chunking and HTML→markdown code, so a chunk healed at answer time is
byte-identical to one the batch build would have written — see the freshness section.

The snapshot under `_corpus/` remains the crawl-time source of truth; the `content`
the DB now stores is a *served copy* of it (see the data-model note below).

## The data model

Everything lives in one wide table, `chunks` (defined in `index/db.py`'s `SCHEMA`).
One row = one heading-granular section (or one size-split *part* of an oversized
section).

| Column | Purpose |
|---|---|
| `id` | `bigserial` surrogate key |
| `chunk_key` | `sha256(url#anchor#heading_path[#partN])`, **unique** — the identity used for upsert, skip-unchanged, purge, and heal |
| `url`, `anchor`, `page_title`, `heading_path`, `source_link` | The pointer — enough to render a live citation and a `[source]` referral without a lookup |
| `library` | `core` / `vision` / `audio` / … — lets retrieval filter or route per question |
| `kind` | `api` / `tutorial` / `guide` — the per-kind retrieval pools key off this |
| `content_hash` | Page/section content hash; drives the incremental skip and the freshness compare |
| `index_version` | Which build wrote the row (answers are stamped with it) |
| `part` | Ordinal within a size-split section; `0` keeps the legacy key format so older rows stay valid |
| `embedding` | `vector(384)` — the bge-small dense vector (cosine) |
| `tsv` | `tsvector` — Postgres full-text index of the same `indexed_text` |
| `content` | Raw section text, **served at answer time** so hydration needs no live fetch |

A second tiny table, `index_meta` (key/value), stores the live `embed_recipe` so the
build can detect a recipe change and force a full re-embed.

**Indexes.** `chunks_embedding_idx` is HNSW with `vector_cosine_ops` (approximate
nearest-neighbour for the dense channel); `chunks_tsv_idx` is a GIN index over the
`tsvector` (the keyword channel); `chunks_url_idx` supports the freshness/hydrate
lookups that fetch all rows for one page. Uniqueness is on `chunk_key` — the whole
upsert/skip/heal machinery hangs off it.

**Width can't drift.** `EMBED_DIMS` is derived from the configured model
(`_MODEL_DIMS` table in `db.py`), reading the *same* env var (`TORCHDOCS_EMBED_MODEL`)
that `embed.py` reads, so the column width and the vectors written into it can never
disagree. An unknown model with no `TORCHDOCS_EMBED_DIMS` override is a loud config
error, not a silently wrong-width table. If the dimension *does* change (a model swap),
`ensure_schema` reads `atttypmod` off `pg_attribute`, sees the mismatch, and drops +
recreates the table — the index is treated as a rebuildable cache, not precious state.

**Pool + runtime-migration-at-pool-open.** There are two ways into Neon, for two
workloads:

- `connect()` — one dedicated connection for the batch build (a single long-running
  writer that commits in checkpoints).
- `get_pool()` — a process-wide `ConnectionPool` (cached with `functools.cache`) for
  the web app, where many concurrent questions each borrow a connection for a couple
  of quick reads. Reconnecting per read costs a TLS handshake (~100–300 ms) that would
  dominate answer time under load and risk exhausting Neon's free-tier connection cap.
  `max_size` (`TORCHDOCS_DB_POOL`, default 8) is kept at or under the plan limit;
  `check=check_connection` validates a connection on checkout so a Neon-side idle
  timeout surfaces as a fresh connection rather than a mid-request query error.

The subtle bit: **`RUNTIME_MIGRATIONS` are applied when the pool opens, not only at
build time.** `create table if not exists` won't touch an existing table, so a column
added after the table first shipped (`part`, `content`) needs an idempotent
`alter table … add column if not exists`. The app SELECTs those columns, and a fresh
deploy can go live *before* the next index build runs `ensure_schema` — so `get_pool()`
runs the same migrations itself, or every search would 500 until the next build. Both
writers (`ensure_schema`) and the reader (`get_pool`) apply the identical idempotent
list.

## The retrieval flow

`retrieve()` in `index/retrieve.py` is the engine behind the agent's `search_docs`
tool. For each content **kind** (`api`, `tutorial`, `guide`) it runs two channels and
fuses them *within the kind*:

- **Dense** (`DENSE_SQL`): `embedding <=> query_vector` cosine order over the HNSW
  index, `pool` candidates.
- **Keyword** (`KEYWORD_SQL`): `tsv @@ plainto_tsquery(...)` ranked by `ts_rank` — this
  rescues exact symbol names (`scaled_dot_product_attention`) that dense similarity
  misses.

The two rankings are merged with **Reciprocal Rank Fusion** (`rrf_merge`,
`score = Σ 1/(k+rank)`). A per-pool relevance gate (`RELEVANCE_GAP = 0.15`) drops
candidates far from *that kind's own* best hit — gated per pool, never against a global
best, so a close tutorial can't set a threshold that filters the entire `api` pool out.
Pools are then interleaved round-robin, strongest pool first (`_interleave_pools`), and
the first `k` win. When the query contains a dotted/underscored identifier a third
**symbol channel** (`SYMBOL_SQL`) runs, matching the token in url/title/heading and
preferring `kind='api'`; it leads the interleave, and an *exact* API-reference hit is
pinned to position 0 — the docs-search behaviour users expect.

Two hard-won details live here as comments:

- **`SET hnsw.ef_search = 150`** on every pool query. HNSW is approximate *and*
  pgvector applies the `WHERE kind=…` filter *after* the index scan; at the default
  `ef_search=40` a `kind='api'` query first collects the ~40 globally-nearest chunks
  (mostly tutorials for a descriptive question) and only then filters — the api page
  can be discarded before the filter sees it. Widening the candidate scan (~2 ms extra
  on a 7K-chunk index) rescues pages whose true in-pool rank is good.
- **The cross-encoder reranker is gone.** A rerank stage sat between fusion and the
  final order until the 2026-07-10 ablation on real content measured it at
  +0.02 recall / −0.005 MRR — no earned keep for a ~90 MB model and per-query cost.
  The fused RRF order is the final order. (Global-RRF, a reference channel, and reserved
  seats were also tried and rejected; see the module docstring.)

**What a "pointer" is.** A pointer is a dict of `POINTER_COLUMNS` — `chunk_key, url,
anchor, page_title, heading_path, library, kind, source_link, part` — **plus `content`**.
That last column is the latency story: `content` rides along in the *same* retrieval
query, so the answer path hydrates each section straight from it with zero extra
round-trips. `hydrate_section` in `index/hydrate.py` takes a fast path when
`pointer["content"]` is present (just returns it); only an empty/un-backfilled row falls
through to a snapshot read or live fetch. Previously each of *k* sections was a separate
live page fetch — the dominant answer latency (measured p50 ≈ 12 s, one outlier 69 s).
`hydrate_sections` also runs the remaining fetch-path work concurrently in a thread pool
(preserving retrieval order, dropping gone sections), collapsing *k* round-trips into
roughly one. `hydrate_page` serves a whole page for `read_page`, returning the heading
outline instead when a page exceeds `PAGE_CHAR_LIMIT` (30 k chars).

`top_distance()` is a small sibling of retrieve: it returns the single smallest cosine
distance for a query (a topicality signal the agent's input guard uses to tell on-topic
from off-topic), using the same pooled access.

## Freshness (stale-while-revalidate)

`index/freshness.py` implements the standard stale-while-revalidate pattern for the
docs index: **serve the stored copy instantly, revalidate right after.** The design
model that drove it: the product promises answers grounded in the docs *as the site
serves them today*, but hydrating from the DB `content` column means an answer can be
served from a chunk that drifted since the last weekly crawl. So *after* an answer goes
out, `refresh_pages(cited_urls)` re-checks exactly the pages that answer cited and heals
any drift, telling the caller which urls changed so it can regenerate the just-shown
answer.

The pass, step by step:

1. **TTL gate** (`_due`, `TORCHDOCS_FRESHNESS_TTL_SECONDS`, default 3600 s). A
   thread-safe check-and-set so one hot page isn't re-fetched on every question; the
   URL table is bounded (evicted past 4096 entries) for a long-lived process. Docs move
   slowly and the weekly crawl is the backstop — an hour bounds both fetch traffic and
   the staleness window.
2. **Live compare.** `_live_units` fetches the page *right now* and chunks it with the
   exact `ingest` code path (`fetch_html` → `extract_main_html` → `to_markdown` →
   `chunk_page`), computing the page-level `content_hash` the same way the crawl does.
   Each live chunk is matched by `chunk_key` to its stored row and compared by `content`.
3. **In-place heal** (`_HEAL`). A drifted row is fixed *completely*: `content`,
   `content_hash`, a freshly computed `embedding`, and the `tsv` — all in one UPDATE. The
   embedding model is already hot in this process (it embeds every incoming query), so
   re-embedding a handful of chunks costs milliseconds and leaves the row exactly as a
   fresh build would have written it. Because the stored hash now matches the re-embedded
   text, the next weekly crawl correctly *skips* the healed rows instead of redoing them.
4. **Scope limit.** The re-embed re-uses the page's *existing* gloss and hypothetical
   questions (via `indexed_text`) — they describe what the symbol is *for*, which small
   edits don't change. Sections that appeared or vanished on the live page are skipped;
   structural change (new/deleted pages, restructured sections, drift drastic enough to
   need fresh enrichment) stays the job of the periodic Build Index crawl.
5. **Fail-open + kill switch.** Every failure — a dead page, a bad fetch, a DB error — is
   logged and skipped; the outer `try/except` guarantees a freshness error can never
   break an already-shown answer. `TORCHDOCS_FRESHNESS=0` disables the pass entirely.

## Design decisions & rationale

- **Local bge-small on CPU (no API, no quota, no cost).** Gemini's free embedding quota
  (~100 items/day) would take *weeks* for the corpus; a 130 MB open model has no key, no
  quota, and no per-item cost — the whole corpus embeds in minutes on a CI runner's CPU,
  and the *same* model embeds queries at answer time (which is what makes millisecond
  in-process healing possible). The tradeoff is accepted and bounded: bge-small is
  English-only, and a controlled A/B against bge-base (768d, full re-embed + benchmark,
  2026-07-08) showed **identical** recall (0.846) and marginally worse MRR at 4× the
  model size and 2× the build time — no measurable gain, so the cheaper model stays. The
  `_MODEL_DIMS` table remains so a future swap is a one-line change plus an automatic
  rebuild.
- **`EMBED_RECIPE` versioning.** `indexed_text()` embeds far more than raw body: symbol
  + synopsis + Contextual-Retrieval gloss + QuOTE-style hypothetical questions + heading
  + content (see `retrieval-gaps-and-improvements.md` for *why* — descriptive questions
  live in a different region of embedding space than terse reference pages). When that
  *shape* changes, the row-skip check would otherwise keep stale vectors, so the recipe
  string (`v7-<model>-g<gloss_stamp>-q<questions_stamp>`) is stored in `index_meta`; any
  change forces a one-time full re-embed. The gloss/question **content stamps** are
  hashes of the committed enrichment files, so editing an enrichment file (or its first
  arrival) forces the re-embed by itself — no manual version bump to forget. Folding the
  model tag in means even a same-dims model swap re-embeds and `index_meta` stays honest
  about which model's vectors are live.
- **`content_hash` skip-unchanged.** A chunk whose `(chunk_key, content_hash)` already
  matches the DB is skipped; every batch commits, so a build is resumable and CI-safe
  (kill it anytime, re-run continues). This is the chat-langchain record-manager lesson.
  Introducing size-capped `part` rows embedded *only* the new parts because `part 0`
  keeps the legacy key format.
- **Why `content` moved into the DB.** Originally the DB stored *no* text — content was
  re-read from the `_corpus/` snapshot (or fetched live) at answer time. On a deployed
  Space with no bundled snapshot, that per-section live fetch was the dominant answer
  latency. Storing `content` lets retrieval hand back the section text in the row it
  already returns; the migration is safe because empty `content` falls through to the old
  fetch path, and `_backfill_content` fills pre-existing rows once with a cheap
  metadata-only UPDATE (no re-embed).
- **pgvector + tsvector over a dedicated vector DB.** Postgres already holds the pointers
  and content; putting the vectors (HNSW) and the keyword index (GIN `tsvector`) in the
  *same* table means a hybrid query is two SELECTs on one connection, an upsert is
  atomic, and freshness heals content + vector + keyword index in one statement. A
  separate vector store would add an operational component, a second consistency problem,
  and cross-store fan-out — for a lean index where "the agent loop, not index
  sophistication, carries retrieval quality" (see the design doc), it isn't worth it.

## Tool & library choices

| Tool | Role | Why |
|---|---|---|
| `psycopg[binary,pool]` (v3) | Neon client + `ConnectionPool` | Modern psycopg with server-side params and a first-party pool; `open=False` + explicit `open()` avoids the deprecated constructor-time open |
| `pgvector` (Postgres extension) | `vector` column + HNSW index + `<=>` cosine | Keeps dense search inside Postgres, next to pointers and content |
| Postgres `tsvector` / GIN | Keyword channel | Free lexical search co-located with the vectors; rescues exact symbol matches dense misses |
| `sentence-transformers` + `BAAI/bge-small-en-v1.5` | Local 384d embeddings (docs + queries) | No API/quota/cost, runs on CPU in CI, hot in-process for query embedding and healing |
| Neon (serverless Postgres) | Managed store | Free tier fits the corpus; connection cap is the reason for the shared pool |

The model is loaded exactly once via a double-checked lock (`_model()`), not
`functools.cache`, so two concurrent first queries can't both load the 130 MB model;
`encode()` itself is safe to call concurrently on the shared instance.

## File by file

- **`__init__.py`** — empty package marker.
- **`db.py`** — Neon connectivity and schema ownership: `EMBED_DIMS` derived from the
  model, the `chunks` + `index_meta` `SCHEMA`, `connect()` (single writer) vs `get_pool()`
  (shared pool with runtime migrations applied at open), `RUNTIME_MIGRATIONS`,
  `ensure_schema()` (idempotent create + migrate + rebuild-on-dims-change), and the
  `get_meta`/`set_meta` recipe helpers.
- **`embed.py`** — the batch index build. `indexed_text()` assembles what actually gets
  embedded and tsvector'd (symbol + synopsis + gloss + questions + heading + body);
  `EMBED_RECIPE` and the enrichment stamps drive full-re-embed decisions; `build_index()`
  walks the snapshot, skips unchanged chunks, upserts new/changed ones in committing
  batches, backfills `content`, purges stale rows, and stamps the recipe. Local model
  loading and `embed_texts`/`embed_query` live here too.
- **`retrieve.py`** — hybrid retrieval: per-kind dense + keyword pools, RRF fusion, the
  symbol channel, per-pool relevance gating and round-robin interleave, `ef_search`
  widening, exact-API pinning. Returns pointers with `content` attached. Also
  `top_distance()` for the input guard.
- **`hydrate.py`** — turns pointers into content: `hydrate_section` (fast path off the
  stored `content`, else snapshot-then-live fallback), `hydrate_sections` (concurrent),
  and `hydrate_page` (whole page, outline-first when oversized). `TORCHDOCS_LIVE_HYDRATE=0`
  requires the snapshot.
- **`freshness.py`** — the post-answer stale-while-revalidate pass: TTL gate, live fetch +
  chunk-by-chunk compare, full in-place heal (content + hash + embedding + tsv), fail-open,
  and the `TORCHDOCS_FRESHNESS=0` kill switch.

## Related docs

- [`../design-content-and-agent-flow.md`](../design-content-and-agent-flow.md) — the
  system design this package implements (corpus, tools, session flow, live links).
- [`../retrieval-gaps-and-improvements.md`](../retrieval-gaps-and-improvements.md) — the
  measured retrieval-quality analysis behind glosses, hypothetical questions, the
  `ef_search` fix, and the reranker decision.
- [`../ingest/README.md`](../ingest/README.md) — the sibling package that builds the
  snapshot this one indexes.
- [`../agent/README.md`](../agent/README.md) — the consumer of the pointers this package
  returns.
