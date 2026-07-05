# TorchDocs Agent — Detailed Execution Plan (TODO level)

Working document for execution. Every task is written so it can be picked up and completed independently, with an **acceptance criterion** ("done when...") and a time estimate.
For the architectural "how" behind these tasks — content extraction cadence, agent access levels, session lifecycle, LangChain vs LangGraph, and live-link mapping — see [docs/design-content-and-agent-flow.md](docs/design-content-and-agent-flow.md).
Tasks marked `[CORE]` are mandatory; `[STRETCH]` — only if time remains. Do not start STRETCH work before all CORE tasks of that milestone are green.

**Binding decisions (do not reopen during execution):**
- Pinned PyTorch version: **torch 2.7.x** — the index, the sandbox, and eval all run on the same version.
- Corpus scope: **only** `torch/nn`, `torch/optim`, `torch/utils/data`, `torch/nn/functional`, `torch/autograd` + their official docs. No C++, no CUDA, no internals.
- **Pointer-based storage:** the DB does not store raw code — only embeddings, tsvector, and metadata (path, lines, symbol, signature). The single source of truth is the pinned clone; content is read from it at query time (hydrate).
- Language: Python 3.11+. All LLM calls go through LiteLLM starting from day one of M3 (before that — direct SDK).
- **Open Knowledge Format (OKF)** — Google's markdown + YAML-frontmatter convention for agent-readable knowledge — is used wherever we hand-author or generate *knowledge documents* consumed by agents or humans: doc chunks (2.1), and all `docs/*.md` reports (hallucinations, error-analysis, loop-vs-langgraph). It is **not** used for the `chunks` DB schema or code chunking: that data is pointer-based (no stored content) and already has its own typed columns, so wrapping it in OKF would add a translation layer with no consumer. Use OKF where it replaces ad-hoc formatting, not where it duplicates an existing schema.

---

## M0 · Setup (1–2 days)

- [ ] [CORE] New repo `torchdocs-agent` with the structure from the README (`ingest/`, `index/`, `agent/`, `eval/`, `app/`), `pyproject.toml`, `ruff`, `pytest`, pre-commit.
  ✔ Done when: `pytest` runs green on a single placeholder test.
- [ ] [CORE] Accounts: Neon (project + DB), at least one LLM key (Anthropic/OpenAI), Langfuse cloud (or defer self-hosting to M4).
  ✔ Done when: `psql $NEON_URL -c "select 1"` works and `.env.example` exists in the repo.
- [ ] [CORE] `scripts/smoke.py`: one LLM call + a write/read against Neon.
  ✔ Done when: the script runs cleanly from the command line.

---

## M1 · The Generation Core (Weeks 1–2)

### 1.1 Output schema
- [ ] [CORE] `agent/schemas.py`: Pydantic model `CodeAnswer` with fields `code: str`, `explanation: str`, `symbols_used: list[str]`, `torch_version: str`.
  ✔ Done when: a round-trip test (dict → model → dict) passes.

### 1.2 LLM wrapper
- [ ] [CORE] `agent/llm.py`: function `generate_code(question: str) -> CodeAnswer` with structured output, retry (up to 3, exponential backoff), and timeout.
  ✔ Done when: 10 different questions return a valid `CodeAnswer` without exceptions.
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
- [ ] [CORE] `ingest/clone.py`: download torch 2.7 (pinned tag, `--depth 1`) and filter to in-scope directories only.
  ✔ Done when: the `_corpus/` directory contains only files from the in-scope modules (hundreds of files, not thousands).
- [ ] [CORE] `ingest/chunk_code.py`: structure-aware code chunking using the `ast` module — one chunk per function/class, with metadata: `file_path`, `start_line`, `end_line`, `symbol_name`, docstring.
  ✔ Done when: running on `torch/nn/modules/linear.py` produces separate chunks for `Linear`, `Bilinear`, etc., with correct line ranges.
- [ ] [CORE] `ingest/chunk_docs.py`: chunk the rst/markdown doc files by heading, same metadata schema. Emit each chunk as an **OKF-style unit**: YAML frontmatter (`file_path`, `start_line`, `end_line`, `symbol_name`, `kind`) over a markdown body, before it's split into DB columns — this is the one point in the pipeline where an intermediate on-disk artifact is worth having, since it's a human/agent-readable knowledge snapshot of the docs corpus, not just a DB-loading step.
  ✔ Done when: a sample of 5 files chunks sensibly under manual review, and the OKF units are valid (frontmatter parses, required keys present).

### 2.2 Indexing in Neon
- [ ] [CORE] Table schema: `chunks(id, embedding vector, tsv tsvector, file_path, start_line, end_line, symbol_name, signature, kind)` — **no raw content column**. The tsvector is computed at index time (from content that is read but not stored) and is sufficient for keyword search. HNSW index on embedding + GIN on tsv.
  ✔ Done when: a migration runs clean, and `select * from chunks limit 1` contains no code — only vectors and metadata.
- [ ] [CORE] `index/embed.py`: compute embeddings in batches (resilient to mid-run failure — checkpointing), and insert into Neon.
  ✔ Done when: the entire corpus is indexed; `count(*)` is sensible; re-running doesn't duplicate rows.

### 2.3 Hybrid retrieval
- [ ] [CORE] `index/retrieve.py`: function `retrieve(query, k=8)` that merges dense (pgvector) + keyword (tsvector) search with simple RRF ranking. Returns **pointers** (path + line range), not content.
  ✔ Done when: searching `scaled_dot_product_attention` returns the pointer to the real definition as the top result (dense alone fails this — that's the test).
- [ ] [CORE] `index/hydrate.py`: read the actual lines from the pinned clone based on the pointers, ready for prompt injection.
  ✔ Done when: hydrating a retrieve result returns exactly the function, and a test confirms the metadata matches the file content.
- [ ] [STRETCH] reranker (small cross-encoder or LLM-rerank) over the top-20.

### 2.4 Wiring and evaluation
- [ ] [CORE] Update `generate_code`: retrieve → hydrate → inject the snippets into the prompt with an explicit instruction "use only APIs that appear in the context", and add `citations: list[{file_path, lines}]` to the schema.
  ✔ Done when: answers include real citations that can be opened in the file.
  *Note: from this point the pinned clone is a runtime dependency — it goes into the deploy image in M5.*
- [ ] [CORE] Dedicated metric `grounded_api_rate`: percentage of symbols in `symbols_used` that exist in the index. Run on the 15 M1 questions, compare before/after RAG.
  ✔ Done when: there is one table showing the improvement — also great material for the README.
- [ ] [STRETCH] RAGAS on the question set (context precision/recall, faithfulness).

**Gate to M3:** `grounded_api_rate` improved significantly over M1, and the hallucinations from `hallucinations.md` are gone or reduced.

---

## M3 · The Agent (Weeks 5–6)

### 3.1 Code execution sandbox
- [ ] [CORE] `agent/runner.py`: run code in an isolated subprocess — Docker image with torch 2.7 **CPU-only** (saves GBs and a GPU machine), 30-second timeout, memory limit, no network.
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
- [ ] [CORE] Cache in Upstash Redis: exact-match on (question, index version) for answers, and a cache for query embeddings.
  ✔ Done when: a repeated question returns from cache in <200ms, and hit-rate is measured.
- [ ] [STRETCH] Semantic cache (vector similarity between questions) — only after the exact cache works.

**Gate to M5:** one complete eval report + one trace that can be shown in an interview.

---

## M5 · Shipping + Hardening (Week 8)

- [ ] [CORE] Minimal Gradio interface: question field, answer with highlighted code, clickable citations, and a "code runs ✓" indicator.
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
