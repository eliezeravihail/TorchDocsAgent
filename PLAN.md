# TorchDocs Agent — Detailed Execution Plan (TODO level)

Working document for execution. Every task is written so it can be picked up and completed independently, with an **acceptance criterion** ("done when...") and a time estimate.
For the architectural "how" behind these tasks — content extraction cadence, agent access levels, session lifecycle, LangChain vs LangGraph, and live-link mapping — see [docs/design-content-and-agent-flow.md](docs/design-content-and-agent-flow.md).
Tasks marked `[CORE]` are mandatory; `[STRETCH]` — only if time remains. Do not start STRETCH work before all CORE tasks of that milestone are green.

**Binding decisions (do not reopen during execution):**
- **Always-latest, no version pinning:** the index tracks what the site serves under `docs.pytorch.org/docs/stable/` (which always points at the latest release) and the current tutorials. A PyTorch release is not a project event — it just shows up as a large hash-diff on the next recrawl and re-embeds automatically. Source referrals point at GitHub `main` (what you see when you open the repo today). `index_version` remains an internal crawl-build id (for cache invalidation and eval comparability), decoupled from PyTorch version numbers. **Non-goal:** answering per the user's installed version — one truth only, what the site and `main` say now.
- **No code execution.** Answers are docs-grounded explanations with illustrative snippets, statically checked (parse, imports, symbols-exist-in-index) but never run. There is no sandbox, no Docker runner, no run-fix loop.
- **Agent-with-tools architecture**, not a fixed pipeline: the LLM iterates over three bounded tools — `search_docs` (hybrid retrieval, repeatable with reformulated queries), `read_page` (whole-page hydrate), `ask_source` (DeepWiki MCP) — until it declares coverage or a budget trips. See `docs/design-content-and-agent-flow.md` §2–3.
- **Source-code questions via DeepWiki, never via our own index**: `ask_source` calls DeepWiki's public MCP server on `pytorch/pytorch`; every claim derived from it carries a referral link (`[source]` on the API page / DeepWiki / GitHub search). If DeepWiki is down, the tool degrades to returning referral links only.
- Corpus scope: the **public documentation site**, tiered (see the seed table in `docs/design-content-and-agent-flow.md` §1.1) — v1 core: API reference (`docs.pytorch.org/docs/{version}`), tutorials, get-started, **torchvision and torchaudio doc sets**; v1.1: the other domain-library doc sets (ExecuTorch, torchao, torchtune, …) added as config-only seeds. **Source code and forums are not indexed**: for implementation questions the agent refers the user out via the `[source]` links captured at crawl time. (Supersedes the earlier five-source-modules scope.)
- **Pointer-based storage:** the DB does not store page text — only embeddings, tsvector, and pointers (`url`, `anchor`, page title, heading path, content hash). The source of truth for the index is the on-disk crawl snapshot; content is read from it at query time (hydrate), and citations link to the live URLs.
- Language: Python 3.11+. All LLM calls go through LiteLLM starting from day one of M3 (before that — direct SDK). **Generation in M1–M2 runs on the Gemini free tier (`gemini-2.5-flash`, `TORCHDOCS_PROVIDER=gemini`)**; switching to a paid provider later is a config change, not code.
- **Embeddings: local open model (`BAAI/bge-small-en-v1.5`, 384 dims) on CPU** — no API, no key, no quota, $0. Decided after Gemini's free embedding quota (~100 items/day) proved unable to index a 7K-chunk corpus. The same model embeds chunks (in CI, minutes) and queries (in-app, ~ms); swapping models is an env var + automatic table rebuild.
- **Open Knowledge Format (OKF)** — Google's markdown + YAML-frontmatter convention for agent-readable knowledge — is used wherever we hand-author or generate *knowledge documents* consumed by agents or humans: doc chunks (2.1), and all `docs/*.md` reports (hallucinations, error-analysis, loop-vs-langgraph). It is **not** used for the `chunks` DB schema or code chunking: that data is pointer-based (no stored content) and already has its own typed columns, so wrapping it in OKF would add a translation layer with no consumer. Use OKF where it replaces ad-hoc formatting, not where it duplicates an existing schema.

---

## M0 · Setup (1–2 days)

- [x] [CORE] New repo `torchdocs-agent` with the structure from the README (`ingest/`, `index/`, `agent/`, `eval/`, `app/`), `pyproject.toml`, `ruff`, `pytest`, pre-commit.
  ✔ Done when: `pytest` runs green on a single placeholder test.
- [x] [CORE] Accounts: Neon (project + DB) and a Gemini API key (Google AI Studio — free tier; covers both generation and embeddings for M1–M2). Anthropic/OpenAI key and Langfuse — optional until M3/M4; the Max consumer subscription does not cover API usage, so a paid provider means loading Console credits separately.
  ✔ Done when: `psql $NEON_URL -c "select 1"` works and `.env.example` exists in the repo.
- [x] [CORE] `scripts/smoke.py`: one LLM call + a write/read against Neon.
  ✔ Done when: the script runs cleanly from the command line.

---

## M1 · The Generation Core (Weeks 1–2)

### 1.1 Output schema
- [x] [CORE] `agent/schemas.py`: Pydantic model `Answer` with fields `answer_md: str` (markdown, may embed code snippets), `symbols_used: list[str]`, `torch_version: str` (citations and referrals join the schema in M2).
  ✔ Done when: a round-trip test (dict → model → dict) passes.

### 1.2 LLM wrapper
- [x] [CORE] `agent/llm.py`: function `answer_question(question: str) -> Answer` with structured output, retry (up to 3, exponential backoff), and timeout.
  ✔ Done when: 10 different questions return a valid `Answer` without exceptions.
- [x] [CORE] Parsing-failure handling: if the output doesn't fit the schema — one repair attempt with the error message, otherwise return a clean error.
  ✔ Done when: a test with a mock that returns broken JSON passes.

### 1.3 First eval — from day one
- [x] [CORE] `eval/checks.py`: three static checks on every `Answer`: (a) every code block in `answer_md` passes `ast.parse`; (b) every `import` in those blocks is torch/standard library; (c) every symbol in `symbols_used` actually appears in the answer.
  ✔ Done when: the checks run on 10 answers and print a pass/fail table.
- [x] [CORE] `eval/questions_v0.jsonl`: 15 manual questions covering the five question types (usage: "how do I use SGD?"; catalog: "what LR schedulers exist?"; recipe: "how do I build a sequence network to detect cats?", "how do I generate music?"; source: "how is conv2d implemented?"; edge: "how do I run a fraud-detection model in the browser?").
  ✔ Done when: the file exists and `eval/run_v0.py` runs all of them and saves results.
- [x] [CORE] **Document hallucinations**: run the 15 questions, manually review the code, and record in `eval/hallucinations.md` every invented API or wrong signature, as an OKF unit (YAML frontmatter with `question_id`, `torch_version`, `severity` + a markdown body per finding).
  ✔ Done when: at least 3 examples are documented. *(This is the measurable justification for M2 — don't skip it.)*

**Gate to M2: ✅ MET (2026-07-05).** Generator works on 3 provider paths; 15/15 v0 answers (poolside/laguna via OpenRouter, on Actions); 5 hallucination findings documented in eval/hallucinations.md.

---

## M2 · Grounding (Weeks 3–4)

### 2.1 Ingestion
- [ ] [CORE] `ingest/discover.py`: enumerate the page list — parse Sphinx `objects.inv` for the API reference (symbol → page + anchor) and the sitemap/toctree for tutorials and get-started; emit a seed-scoped URL list.
  ✔ Done when: the list covers the `docs/stable` API tree, tutorials, and the torchvision/torchaudio doc sets (thousands of URLs, not tens of thousands), and `torch.nn.Linear` maps to its exact page + anchor.
- [ ] [CORE] `ingest/crawl.py`: fetch rendered pages, strip nav/chrome, convert HTML → markdown, and save to the `_corpus/` snapshot with per-page metadata (`url`, `title`, `section_path`, `content_hash`, crawl date). Idempotent: unchanged `content_hash` → skip.
  ✔ Done when: a re-run over an unchanged site fetches but re-processes ~0 pages, and 5 sampled pages read cleanly as markdown.
- [ ] [CORE] `ingest/chunk_docs.py`: chunk each snapshot page by heading — a chunk is one section, code blocks stay attached to their section, and API pages record their `[source]` GitHub link as metadata. Emit each chunk as an **OKF-style unit**: YAML frontmatter (`url`, `anchor`, `page_title`, `heading_path`, `kind`) over a markdown body — a human/agent-readable knowledge snapshot of the docs corpus, not just a DB-loading step.
  ✔ Done when: a sample of 5 pages chunks sensibly under manual review, and the OKF units are valid (frontmatter parses, required keys present).

### 2.2 Indexing in Neon
- [ ] [CORE] Table schema: `chunks(id, embedding vector, tsv tsvector, url, anchor, page_title, heading_path, library, source_link, kind, content_hash, index_version)` — **no raw content column**. The tsvector is computed at index time (from content that is read but not stored) and is sufficient for keyword search. HNSW index on embedding + GIN on tsv; unique on `(url, anchor, index_version)`.
  ✔ Done when: a migration runs clean, and `select * from chunks limit 1` contains no page text — only vectors and metadata.
- [ ] [CORE] `index/embed.py`: compute embeddings with `gemini-embedding-001` in rate-limit-aware batches (respect free-tier RPM/TPM with backoff; resilient to mid-run failure — checkpointing), and upsert into Neon; unchanged `content_hash` → skip (this is what makes the weekly recrawl cheap).
  ✔ Done when: the entire corpus is indexed; `count(*)` is sensible; re-running over an unchanged snapshot embeds 0 chunks; hitting a mocked 429 backs off instead of crashing.

### 2.3 Hybrid retrieval
- [ ] [CORE] `index/retrieve.py`: function `retrieve(query, k=8)` that merges dense (pgvector) + keyword (tsvector) search with simple RRF ranking. Returns **pointers** (`url` + `anchor`), not content.
  ✔ Done when: searching `scaled_dot_product_attention` returns the pointer to its API-reference section as the top result (dense alone fails this — that's the test).
- [ ] [CORE] `index/hydrate.py`: read content from the crawl snapshot — section-level (for `search_docs` results) and whole-page (for `read_page`), with an outline-first guardrail for oversized pages.
  ✔ Done when: hydrating a retrieve result returns exactly the section, a whole-page hydrate returns clean markdown, and a test confirms the metadata matches the snapshot content.
- [ ] [STRETCH] reranker (small cross-encoder or LLM-rerank) over the top-20.

### 2.4 Wiring and evaluation
- [ ] [CORE] Update `answer_question`: retrieve → hydrate → inject the sections into the prompt with an explicit instruction "answer only from the provided context; if it's not there, say so and refer via the `[source]`/search link", and add `citations: list[{url, anchor, page_title}]` + `referrals: list[{url, reason}]` to the schema.
  ✔ Done when: answers include real citations that open in a browser on the exact section.
  *Note: from this point the crawl snapshot is a runtime dependency — it goes into the deploy image (or a mounted volume) in M5.*
- [ ] [CORE] Dedicated metric `grounded_api_rate`: percentage of symbols in `symbols_used` that exist in the index. Run on the 15 M1 questions, compare before/after RAG.
  ✔ Done when: there is one table showing the improvement — also great material for the README.
- [ ] [STRETCH] RAGAS on the question set (context precision/recall, faithfulness).

**Gate to M3:** `grounded_api_rate` improved significantly over M1, and the hallucinations from `hallucinations.md` are gone or reduced.

---

## M3 · The Agent (Weeks 5–6)

### 3.1 The three tools
- [x] [CORE] `agent/tools.py`: `search_docs(query, library=None)` (wraps retrieve+hydrate, returns pointers + section text), `read_page(url)` (whole-page hydrate with outline-first guardrail), `ask_source(question)` (DeepWiki MCP client on `pytorch/pytorch` + domain repos; on failure returns referral links only). Each tool result is a typed dict the LLM sees verbatim.
  ✔ Done when: each tool has a unit test, and `ask_source` with the network mocked-down still returns usable referral links.

### 3.2 The manual tool loop
- [x] [CORE] `agent/loop.py`: manual tool-calling loop (~100 lines target): the LLM iterates over the three tools within budgets (`search_docs` ≤6, `read_page` ≤2, `ask_source` ≤1) until it declares coverage → generate structured `Answer` → static checks → citations + referrals.
  ✔ Done when: "how do I generate music?" produces ≥2 distinct `search_docs` queries and an answer citing several pages; "how do I use SGD?" resolves in a single search; budget exhaustion yields an honest gap answer, not a bluff.
- [x] [CORE] Static-check regeneration: if `eval/checks.py` fails (unparseable snippet, symbol not in index, `ask_source` claim without a referral), regenerate once with the specific failures injected; a second failure delivers the answer with a visible warning.
  ✔ Done when: a mocked hallucinated symbol triggers exactly one regeneration round.
- [x] [CORE] Source-question path: "how is conv2d implemented?" flows docs-first, then `ask_source`, and the answer separates docs-cited content from DeepWiki-derived content with a referral link.
  ✔ Done when: the answer renders the distinction and the referral URL resolves.

### 3.3 LiteLLM gateway
- [ ] [CORE] Route all calls through the LiteLLM proxy with config: primary provider + fallback, daily budget, and tag every call (`m3-loop`, `m3-tool-search`...).
  ✔ Done when: a per-request cost report appears in the LiteLLM logs.

### 3.4 LangGraph and comparison
- [x] [CORE] Rewrite the loop as a LangGraph graph (the exact same nodes).
  ✔ Done when: both versions pass the same 15-question set with similar results.
- [x] [CORE] `docs/loop-vs-langgraph.md`: short comparison — lines of code, ease of debugging, latency. One page, as an OKF unit (YAML frontmatter with `compared` and `date`).
- [ ] [STRETCH] Expose `search_docs` as an MCP server with FastMCP; test from an MCP client.
- [ ] [STRETCH] Parallel tool fan-out (several `search_docs` calls concurrently in the LangGraph version).
- [ ] [STRETCH] Long-term memory (user preferences, torch version) — defer if no time.

**Gate to M4:** a real recipe question and a real source question both go through the tool loop end to end with correct citations/referrals.

---

## M4 · Discipline (Week 7)

- [ ] [CORE] Wire up Langfuse: a trace for every run with a span per tool call plus generate/check spans.
  ✔ Done when: a failed run can be opened in the UI and you can see which tool call or step broke, and what each tool returned.
- [ ] [CORE] Expand the eval set to **40 questions** in `eval/questions_v1.jsonl`, each with: question, type (usage/catalog/recipe/source/edge), and an automatic assertion (e.g. "the answer must mention ≥5 scheduler classes", "must cite the DataLoader page", "must include a referral, not a fabricated answer").
  ✔ Done when: `eval/run_v1.py` runs all of them and prints: pass rate, grounded_api_rate, citation-validity rate, average cost and latency.
- [ ] [CORE] Error taxonomy: classify every failure into one of 4 categories (fake API / missed retrieval / bluffed instead of referring out / wrong citation), log it in MLflow, and write `docs/error-analysis.md` (OKF unit: frontmatter with `category`, `count`, `eval_version`) with 3 conclusions and one improvement actually implemented.
  ✔ Done when: there is a measurable before/after for at least one improvement.
- [ ] [CORE] Cache in Upstash Redis: exact-match on (question, index version) for answers, and a cache for query embeddings.
  ✔ Done when: a repeated question returns from cache in <200ms, and hit-rate is measured.
- [ ] [STRETCH] Semantic cache (vector similarity between questions) — only after the exact cache works.

**Gate to M5:** one complete eval report + one trace that can be shown in an interview.

---

## M5 · Shipping + Hardening (Week 8)

- [ ] [CORE] Minimal Gradio interface: question field, markdown answer with highlighted snippets, clickable citations (page › section), and visually distinct referral links.
- [ ] [CORE] Deploy on a free tier — pick one: HF Spaces (fastest), Modal, or Railway.
  ✔ Done when: a public link works from a clean browser, including a full query.
- [ ] [CORE] Scheduled recrawl: a weekly job (cron / GitHub Action) that runs discover → crawl → embed incrementally (hash-diff), bumps `index_version` only when content changed, and logs how many pages changed.
  ✔ Done when: two consecutive runs against an unchanged site produce a "0 pages changed" log line and no new rows.
- [ ] [CORE] `ingest/watch.py` — release watcher: a daily job that polls the GitHub Releases API of `pytorch/pytorch`; a new stable tag immediately kicks the recrawl job instead of waiting for the weekly slot (the recrawl itself handles everything via hash-diff). Watches **releases, not commits** — commits are noise relative to the rendered docs site.
  ✔ Done when: pointing it at a mocked "new release" response triggers a recrawl; a normal day produces a single "no new release" log line.
- [ ] [CORE] Basic auth: an API key per user (table in Neon), rate limit per key, and every request tagged to a key.
  ✔ Done when: a request without a key is rejected; one key cannot exceed its quota.
  *Note: not full OAuth. API keys are enough to demonstrate multi-user support.*
- [ ] [CORE] Cost ceilings: per-key budget and a global daily cap via LiteLLM; exceeding it returns a clean error.
- [ ] [CORE] Secrets in Infisical (or the deploy platform's secrets manager) — zero secrets in code.
- [ ] [CORE] Update the README: screenshots, an eval results table, a live link.
  ✔ Done when: a stranger can understand the project and try it within 2 minutes.
- [ ] [STRETCH] "Cost story" page: how much an average query costs, and where the free tier runs out.

---

## Future extensions (out of scope for the 8-week plan)
- Ingesting a traceback/diagram screenshot (VLM/OCR).
- A second corpus: libtorch C++ or the docs-site JS.
- WhatsApp/Slack as an additional frontend (same agent, channel wrapper).

## Stop rules
- Stuck for more than half a day on a CORE task? Cut its scope and document the cut — don't extend the time.
- Every milestone closes with a tagged commit (`m1-done`...) and a summary line in the README.
- Don't touch STRETCH while CORE is red. Don't add features that aren't on the list.
