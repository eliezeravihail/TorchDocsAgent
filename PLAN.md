# TorchDocs Agent — Detailed Execution Plan (TODO level)

Working document for execution. Every task is written so it can be picked up and completed independently, with an **acceptance criterion** ("done when...") and a time estimate.
Tasks marked `[CORE]` are mandatory; `[STRETCH]` — only if time remains. Do not start STRETCH work before all CORE tasks of that milestone are green.

**Binding decisions (do not reopen during execution):**
- **No pinned PyTorch version.** We always track torch's latest default branch (`main`) — `ingest/clone.py` re-clones/updates it on each ingest run, and the sandbox always installs whatever is current. `torch_version` in `CodeAnswer` is informational metadata (which version was current when a chunk was ingested / when code ran), never a compatibility gate to design around.
- Corpus scope: **only** `torch/nn`, `torch/optim`, `torch/utils/data`, `torch/nn/functional`, `torch/autograd` + their official docs. No C++, no CUDA, no internals.
- **Pointer-based storage:** the DB does not store raw code — only tsvector and metadata (path, lines, symbol, signature). The single source of truth is the tracked clone (always latest `main`); content is read from it at query time (hydrate).
- **No vector embeddings / pgvector.** Considered and rejected as the primary retrieval mechanism. Evidence: production agentic coders (Claude Code, Cursor, Devin) and Augment's SWE-bench-winning agent all found grep/structural search over vector similarity for code — code has exact, distinctive symbol names, and an iterative agent can reformulate and retry, which a one-shot embedding lookup can't ([Augment/jxnl writeup](https://jxnl.co/writing/2025/09/11/why-grep-beat-embeddings-in-our-swe-bench-agent-lessons-from-augment/), [why coding agents use grep, not vectors](https://www.mindstudio.ai/blog/is-rag-dead-what-ai-agents-use-instead)). The "understands the fuzzy question" job that embeddings are usually there for is instead done by the LLM itself: it turns a fuzzy question into concrete search terms (symbol names, keywords), searches, and — per 3.2 — rewrites and retries if the result is insufficient. Retrieval is therefore **keyword/structural search only** (Postgres `tsvector` full-text + `pg_trgm` fuzzy matching on symbol names), never dense/vector similarity. Don't reintroduce pgvector unless eval data (M4) actually shows keyword search failing on a real class of questions.
- Language: Python 3.11+. All LLM calls go through LiteLLM starting from day one of M3 (before that — direct SDK).
- **Open Knowledge Format (OKF)** — Google's markdown + YAML-frontmatter convention for agent-readable knowledge — is used wherever we hand-author or generate *knowledge documents* consumed by agents or humans: doc chunks (2.1), and all `docs/*.md` reports (hallucinations, error-analysis, loop-vs-langgraph). It is **not** used for the `chunks` DB schema or code chunking: that data is pointer-based (no stored content) and already has its own typed columns, so wrapping it in OKF would add a translation layer with no consumer. Use OKF where it replaces ad-hoc formatting, not where it duplicates an existing schema.
- **License headers:** the repo is Apache License 2.0. Every `.py` file carries the Apache boilerplate notice (copyright + license pointer) at the top, per the license's own Appendix. New source files must include it from creation — don't add it retroactively as cleanup.
- **PyTorch source license:** torch is licensed under a Modified BSD (BSD-3-Clause style) license, copyright Meta/Facebook Inc. and contributors (see [pytorch/pytorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) and [NOTICE](https://github.com/pytorch/pytorch/blob/main/NOTICE)). Any actual torch source text we serve back to a user — hydrated snippets, citations, quoted excerpts — must carry a license attribution beneath it. This is a display/serving requirement, separate from our own Apache-2.0 headers on our own code.
- **LLM provider:** primary is **Gemini Flash** (Google) — has a genuine free tier (rate-limited, no card required), used for all development and iteration in M0/M1 to avoid burning paid credits before the pipeline is stable. **Claude Haiku** (Anthropic, paid, no persistent free tier) is the comparison/fallback provider — used to sanity-check output quality once the eval set exists, and becomes the official LiteLLM fallback from M3.

---

## M0 · Setup (1–2 days)

- [ ] [CORE] New repo `torchdocs-agent` with the structure from the README (`ingest/`, `index/`, `agent/`, `eval/`, `app/`), `pyproject.toml`, `ruff`, `pytest`, pre-commit.
  ✔ Done when: `pytest` runs green on a single placeholder test.
- [ ] [CORE] Accounts: Neon (project + DB), a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) (free tier), Langfuse cloud (or defer self-hosting to M4).
  ✔ Done when: `psql $NEON_URL -c "select 1"` works and `.env.example` exists in the repo.
- [ ] [CORE] `scripts/smoke.py`: one Gemini Flash call + a write/read against Neon.
  ✔ Done when: the script runs cleanly from the command line, with zero API cost.

---

## M1 · The Generation Core (Weeks 1–2)

### 1.1 Output schema
- [ ] [CORE] `agent/schemas.py`: Pydantic model `CodeAnswer` with fields `code: str`, `explanation: str`, `symbols_used: list[str]`, `torch_version: str`.
  ✔ Done when: a round-trip test (dict → model → dict) passes.

### 1.2 LLM wrapper
- [ ] [CORE] `agent/llm.py`: function `generate_code(question: str) -> CodeAnswer` with structured output, retry (up to 3, exponential backoff), and timeout. Built against **Gemini Flash** first (free tier) so iteration during development costs nothing.
  ✔ Done when: 10 different questions return a valid `CodeAnswer` without exceptions.
- [ ] [STRETCH] Run the same 10 questions through **Claude Haiku** and compare output quality/structured-output reliability — informs which provider becomes the LiteLLM primary in M3.
- [ ] [CORE] Parsing-failure handling: if the output doesn't fit the schema — one repair attempt with the error message, otherwise return a clean error.
  ✔ Done when: a test with a mock that returns broken JSON passes.

### 1.3 First eval — from day one
- [ ] [CORE] `eval/checks.py`: three checks on every `CodeAnswer`: (a) `ast.parse` succeeds; (b) every `import` is torch/standard library; (c) every symbol in `symbols_used` actually appears in the code.
  ✔ Done when: the checks run on 10 answers and print a pass/fail table.
- [ ] [CORE] `eval/questions_v0.jsonl`: 15 manual PyTorch questions (5 easy: "what does nn.Dropout do"; 5 medium: "write a DataLoader with a custom sampler"; 5 hard: "custom autograd Function").
  ✔ Done when: the file exists and `eval/run_v0.py` runs all of them and saves results.
- [ ] [CORE] **Document hallucinations**: run the 15 questions, manually review the code, and record in `eval/hallucinations.md` every invented API or wrong signature, as an OKF unit (YAML frontmatter with `question_id`, `torch_version`, `severity` + a markdown body per finding).
  ✔ Done when: at least 3 examples are documented. *(This is the measurable justification for M2 — don't skip it.)*

**Gate to M2:** a working generator + the 15-question set + a documented hallucination list.

---

## M2 · Grounding (Weeks 3–4)

### 2.1 Ingestion
- [ ] [CORE] `ingest/clone.py`: clone torch's `main` branch (`--depth 1`), re-running it re-fetches the latest `main` (no fixed tag), and filter to in-scope directories only. Also keep torch's own `LICENSE` and `NOTICE` files from the clone (don't filter them out) — they're the source of truth for the attribution text added in 2.4. Record the resolved commit SHA somewhere (e.g. a `_corpus/COMMIT` file) purely so we know what's currently ingested.
  ✔ Done when: the `_corpus/` directory contains only files from the in-scope modules (hundreds of files, not thousands), plus the top-level `LICENSE`/`NOTICE`/`COMMIT`, and re-running the script updates them in place.
- [ ] [CORE] `ingest/chunk_code.py`: structure-aware code chunking using the `ast` module — one chunk per function/class, with metadata: `file_path`, `start_line`, `end_line`, `symbol_name`, docstring.
  ✔ Done when: running on `torch/nn/modules/linear.py` produces separate chunks for `Linear`, `Bilinear`, etc., with correct line ranges.
- [ ] [CORE] `ingest/chunk_docs.py`: chunk the rst/markdown doc files by heading, same metadata schema. Emit each chunk as an **OKF-style unit**: YAML frontmatter (`file_path`, `start_line`, `end_line`, `symbol_name`, `kind`) over a markdown body, before it's split into DB columns — this is the one point in the pipeline where an intermediate on-disk artifact is worth having, since it's a human/agent-readable knowledge snapshot of the docs corpus, not just a DB-loading step.
  ✔ Done when: a sample of 5 files chunks sensibly under manual review, and the OKF units are valid (frontmatter parses, required keys present).

### 2.2 Indexing in Neon
- [ ] [CORE] Table schema: `chunks(id, tsv tsvector, file_path, start_line, end_line, symbol_name, signature, kind)` — **no raw content column, no vector column**. The tsvector is computed at index time (from content that is read but not stored) over `symbol_name` + `signature` + docstring/heading text. GIN index on tsv; trigram (`pg_trgm`) index on `symbol_name` for fuzzy/typo-tolerant exact-name lookups.
  ✔ Done when: a migration runs clean, and `select * from chunks limit 1` contains no code and no vectors — only text-search metadata.
- [ ] [CORE] `index/load.py`: bulk-insert all chunks' metadata into Neon (no embedding step — this is a plain load, not a compute-heavy batch job).
  ✔ Done when: the entire corpus is indexed; `count(*)` is sensible; re-running doesn't duplicate rows.

### 2.3 Structural + keyword retrieval (no vector search)
- [ ] [CORE] `index/retrieve.py`: function `retrieve(query, k=8)` that runs `tsvector` full-text search plus `pg_trgm` fuzzy matching on symbol names, ranked by Postgres's native rank + trigram similarity. Returns **pointers** (path + line range), not content. No dense/embedding step anywhere in this path.
  ✔ Done when: searching `scaled_dot_product_attention` returns the pointer to the real definition as the top result, and a misspelled `scaled_dot_product_attetion` still finds it via trigram similarity.
- [ ] [CORE] `agent/query_terms.py`: small LLM call that turns a fuzzy natural-language question into 1-3 concrete search terms (candidate symbol names / keywords) to feed `retrieve`. This is the "semantic understanding" step — done by the model, not by a vector index.
  ✔ Done when: "how do I randomly drop out some activations during training" resolves to search terms that include `dropout`/`Dropout` and `retrieve` finds `nn.Dropout`.
- [ ] [CORE] `index/hydrate.py`: read the actual lines from the tracked clone based on the pointers, ready for prompt injection. Each hydrated result carries a fixed `license: "BSD-3-Clause (PyTorch), Copyright (c) Meta Platforms, Inc. and affiliates"` string alongside the content — this is metadata, not something computed per-file, since the whole corpus shares one license.
  ✔ Done when: hydrating a retrieve result returns exactly the function plus its license string, and a test confirms the metadata matches the file content.
- [ ] [STRETCH] LLM-rerank over the top-20 keyword/trigram results, if eval data shows ranking (not recall) is the bottleneck.

### 2.4 Wiring and evaluation
- [ ] [CORE] Update `generate_code`: retrieve → hydrate → inject the snippets into the prompt with an explicit instruction "use only APIs that appear in the context", and add `citations: list[{file_path, lines, license}]` to the schema (the `license` field is the fixed PyTorch attribution string from `hydrate`, not model-generated).
  ✔ Done when: answers include real citations that can be opened in the file, each with a license string attached.
  *Note: from this point the tracked clone is a runtime dependency — it goes into the deploy image in M5.*
- [ ] [CORE] Dedicated metric `grounded_api_rate`: percentage of symbols in `symbols_used` that exist in the index. Run on the 15 M1 questions, compare before/after RAG.
  ✔ Done when: there is one table showing the improvement — also great material for the README.
- [ ] [STRETCH] RAGAS on the question set (context precision/recall, faithfulness).

**Gate to M3:** `grounded_api_rate` improved significantly over M1, and the hallucinations from `hallucinations.md` are gone or reduced.

---

## M3 · The Agent (Weeks 5–6)

### 3.1 Code execution sandbox
- [ ] [CORE] `agent/runner.py`: run code in an isolated subprocess — Docker image with latest torch **CPU-only** (saves GBs and a GPU machine), 30-second timeout, memory limit, no network.
  ✔ Done when: valid code returns stdout; an infinite loop is killed by the timeout; `import requests` fails.
  *Note: start locally with Docker. Moving to Modal — only in M5.*

### 3.2 The manual loop
- [ ] [CORE] `agent/loop.py`: manual agent loop (~150 lines target): plan → retrieve → generate → run → on error: inject the traceback back and fix (up to 3 rounds) → answer with citations.
  ✔ Done when: "build a training loop with mixed precision" goes through the full path and returns code that runs.
- [ ] [CORE] Self-grading on retrieval: after retrieve, a short LLM call judges whether the context is sufficient; if not — rewrite the query and retry once.
  ✔ Done when: there is a test with an ambiguous question that demonstrates query rewriting.

### 3.3 LiteLLM gateway
- [ ] [CORE] Route all calls through the LiteLLM proxy with config: primary provider + fallback, daily budget, and tag every call (`m3-loop`, `m3-grade`...).
  ✔ Done when: a per-request cost report appears in the LiteLLM logs.

### 3.4 LangGraph and comparison
- [ ] [CORE] Rewrite the loop as a LangGraph graph (the exact same nodes).
  ✔ Done when: both versions pass the same 15-question set with similar results.
- [ ] [CORE] `docs/loop-vs-langgraph.md`: short comparison — lines of code, ease of debugging, latency. One page, as an OKF unit (YAML frontmatter with `compared` and `date`).
- [ ] [STRETCH] Expose retrieve + runner as MCP servers with FastMCP; test from an MCP client.
- [ ] [STRETCH] Routing between an "explain" path (no runner) and a "build" path (with runner).
- [ ] [STRETCH] Long-term memory (user preferences, torch version) — defer if no time.

**Gate to M4:** a real build request goes through plan→retrieve→generate→run→fix→cite end to end.

---

## M4 · Discipline (Week 7)

- [ ] [CORE] Wire up Langfuse: a trace for every run with a span per step (plan / retrieve / generate / run / fix).
  ✔ Done when: a failed run can be opened in the UI and you can see at which step it broke.
- [ ] [CORE] Expand the eval set to **40 questions** in `eval/questions_v1.jsonl`, each with: question, type (explain/build), and a gold answer or automatic assertion (e.g. "the code must run a forward pass on a 2x3 tensor without an exception").
  ✔ Done when: `eval/run_v1.py` runs all of them and prints: pass rate, grounded_api_rate, executability rate, average cost and latency.
- [ ] [CORE] Error taxonomy: classify every failure into one of 4 categories (fake API / missed retrieval / runtime error / wrong citation), log it in MLflow, and write `docs/error-analysis.md` (OKF unit: frontmatter with `category`, `count`, `eval_version`) with 3 conclusions and one improvement actually implemented.
  ✔ Done when: there is a measurable before/after for at least one improvement.
- [ ] [CORE] Cache in Upstash Redis: exact-match on (question, index version) for answers, and a cache for extracted search terms (2.3) per question.
  ✔ Done when: a repeated question returns from cache in <200ms, and hit-rate is measured.
- [ ] [STRETCH] Semantic cache (embedding similarity between questions, to catch near-duplicate phrasing) — only after the exact cache works, and only if eval data shows enough near-duplicate traffic to justify it.

**Gate to M5:** one complete eval report + one trace that can be shown in an interview.

---

## M5 · Shipping + Hardening (Week 8)

- [ ] [CORE] Minimal Gradio interface: question field, answer with highlighted code, clickable citations (each showing its `license` string beneath it), and a "code runs ✓" indicator.
- [ ] [CORE] Deploy on a free tier — pick one: HF Spaces (fastest), Modal (more impressive, includes the sandbox), or Railway.
  ✔ Done when: a public link works from a clean browser, including a full query.
- [ ] [CORE] Basic auth: an API key per user (table in Neon), rate limit per key, and every code run tagged to a key.
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
