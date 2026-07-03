# TorchDocs Agent — Detailed Execution Plan (TODO level)

Working document for execution. Every task is written so it can be picked up and completed independently, with an **acceptance criterion** ("done when...") and a time estimate.
Tasks marked `[CORE]` are mandatory; `[STRETCH]` — only if time remains. Do not start STRETCH work before all CORE tasks of that milestone are green.

**What this agent is:** a question-answering and citation agent over PyTorch's public API. It explains, references, and quotes existing code/docs. **It never writes or generates new code.** Every code-shaped thing a user sees is either an exact quote from the real source/docs (with a citation) or doesn't appear at all.

**Binding decisions (do not reopen during execution):**
- **No pinned PyTorch version.** We always track torch's latest default branch (`main`) — `ingest/clone.py` re-clones/updates it on each ingest run. `torch_version` in `Answer` is informational metadata (which version was current when a chunk was ingested), never a compatibility gate to design around.
- **Scope: public API surface only, not internals — for both languages.**
  - **Python**: any public, `ast`-parseable module of torch's public API (`torch.nn`, `torch.optim`, `torch.utils.data`, `torch.nn.functional`, `torch.autograd`, and others as they come up) + their official docs. This is not a hardcoded 5-module allowlist — any pure-Python public API is fair game, since the scope-limiting factor is "must be `ast`-parseable public API," not an arbitrary list.
  - **C++ (libtorch)**: the public C++ API is in scope, but **only via its official documentation** (`pytorch/cppdocs`), not the C++ source. We never parse or chunk C++/CUDA source — no C++ AST tooling needed, because we're not generating or executing C++ either. C++ chunks are `kind='doc'`, tagged `api_lang='cpp'`.
  - **Out of scope, both languages:** implementation internals (`aten` kernels, `c10`, `torch/csrc` bindings, CUDA kernels, build system, tests). Nothing here is user-facing API, so there's nothing to explain or cite from it.
- **No code generation, ever.** The agent explains and cites; it does not synthesize new code. Any snippet shown to a user must be a verbatim substring of a hydrated source/doc chunk, always paired with a citation. This eliminates the code-execution sandbox, the run→fix self-correction loop, and any eval check about whether generated code runs — replaced by a much simpler, more central check: **does every quoted snippet actually appear verbatim in the cited location?**
- **Pointer-based storage:** the DB does not store raw code — only tsvector, (for `kind='doc'`) an embedding, and metadata (path, lines, symbol, signature, kind, api_lang). The single source of truth is the tracked clone (Python) / cppdocs clone (C++); content is read from it at query time (hydrate).
- **Retrieval is split by content kind — not all-or-nothing on embeddings.** Two documented positions turned out to both be right, for different content: (a) production agentic coders (Claude Code, Cursor, Devin) and Augment's SWE-bench-winning agent found grep/structural search beats vector similarity **for code**, since code has exact, distinctive symbol names and an iterative agent can reformulate and retry, which a one-shot embedding lookup can't ([Augment/jxnl writeup](https://jxnl.co/writing/2025/09/11/why-grep-beat-embeddings-in-our-swe-bench-agent-lessons-from-augment/)); (b) mainstream 2026 RAG-chatbot guidance says hybrid (vector + keyword, ~70/30, plus a cross-encoder reranker when precision is low) is standard **for prose documentation**, since keyword alone misses semantically-different phrasing and there's typically no iterative retry loop ([Supermemory](https://supermemory.ai/blog/how-to-build-rag-based-chatbot-guide), [Docsie](https://www.docsie.io/blog/articles/rag-chatbot-enterprise-docs-2026/)). We have both kinds of content, so: **`kind='code'` chunks → keyword/trigram only** (matches (a): exact symbol names, agent retries). **`kind='doc'` chunks → hybrid** (matches (b): prose, including C++ API reference pages, embedding + tsvector + RRF, reranker if eval shows low precision). This is a per-`kind` policy, not a global one — don't collapse it back to one mechanism for both.
- Language: Python 3.11+. All LLM calls go through LiteLLM starting from day one of M3 (before that — direct SDK).
- **Open Knowledge Format (OKF)** — Google's markdown + YAML-frontmatter convention for agent-readable knowledge — is used wherever we hand-author or generate *knowledge documents* consumed by agents or humans: doc chunks (2.1), and all `docs/*.md` reports (hallucinations, error-analysis, loop-vs-langgraph). It is **not** used for the `chunks` DB schema or code chunking: that data is pointer-based (no stored content) and already has its own typed columns, so wrapping it in OKF would add a translation layer with no consumer. Use OKF where it replaces ad-hoc formatting, not where it duplicates an existing schema.
- **License headers:** the repo is Apache License 2.0. Every `.py` file carries the Apache boilerplate notice (copyright + license pointer) at the top, per the license's own Appendix. New source files must include it from creation — don't add it retroactively as cleanup.
- **PyTorch source license:** torch is licensed under a Modified BSD (BSD-3-Clause style) license, copyright Meta/Facebook Inc. and contributors (see [pytorch/pytorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) and [NOTICE](https://github.com/pytorch/pytorch/blob/main/NOTICE)). `pytorch/cppdocs` carries its own license notice too. Any actual torch source/doc text we serve back to a user — hydrated snippets, citations, quoted excerpts — must carry a license attribution beneath it. This is a display/serving requirement, separate from our own Apache-2.0 headers on our own code.
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

## M1 · The Answering Core (Weeks 1–2)

### 1.1 Output schema
- [ ] [CORE] `agent/schemas.py`: Pydantic model `Answer` with fields `explanation: str`, `symbols_referenced: list[str]`, `torch_version: str`. Deliberately **no `code` field** — there is nothing to generate yet, and adding one would invite the model to fill it with invented code.
  ✔ Done when: a round-trip test (dict → model → dict) passes.

### 1.2 LLM wrapper
- [ ] [CORE] `agent/llm.py`: function `answer_question(question: str) -> Answer` with structured output, retry (up to 3, exponential backoff), and timeout. System prompt explicitly instructs: explain in prose, name the real APIs involved in `symbols_referenced`, **never emit a code block**. Built against **Gemini Flash** first (free tier) so iteration during development costs nothing.
  ✔ Done when: 10 different questions return a valid `Answer` without exceptions.
- [ ] [STRETCH] Run the same 10 questions through **Claude Haiku** and compare output quality/structured-output reliability — informs which provider becomes the LiteLLM primary in M3.
- [ ] [CORE] Parsing-failure handling: if the output doesn't fit the schema — one repair attempt with the error message, otherwise return a clean error.
  ✔ Done when: a test with a mock that returns broken JSON passes.

### 1.3 First eval — from day one
- [ ] [CORE] `eval/checks.py`: checks on every `Answer`: (a) `explanation` contains no fenced code block (` ``` `) — a guardrail against the model writing code anyway; (b) `symbols_referenced` is non-empty whenever the question names a specific API.
  ✔ Done when: the checks run on 10 answers and print a pass/fail table.
- [ ] [CORE] `eval/questions_v0.jsonl`: 15 manual PyTorch questions (5 easy: "what does nn.Dropout do"; 5 medium: "what does DataLoader's sampler argument control"; 5 hard: "how does a custom autograd Function's backward get called").
  ✔ Done when: the file exists and `eval/run_v0.py` runs all of them and saves results.
- [ ] [CORE] **Document hallucinations**: run the 15 questions, manually review the explanations, and record in `eval/hallucinations.md` every invented symbol name or wrong claim about API behavior, as an OKF unit (YAML frontmatter with `question_id`, `torch_version`, `severity` + a markdown body per finding).
  ✔ Done when: at least 3 examples are documented. *(This is the measurable justification for M2 — don't skip it.)*

**Gate to M2:** a working answerer + the 15-question set + a documented hallucination list.

---

## M2 · Grounding (Weeks 3–4)

### 2.1 Ingestion
- [ ] [CORE] `ingest/clone.py`: clone torch's `main` branch (`--depth 1`), re-running it re-fetches the latest `main` (no fixed tag), and filter to `ast`-parseable public-API directories only (no C++/CUDA/internals). Also keep torch's own `LICENSE` and `NOTICE` files from the clone (don't filter them out) — they're the source of truth for the attribution text added in 2.4. Record the resolved commit SHA somewhere (e.g. a `_corpus/COMMIT` file) purely so we know what's currently ingested.
  ✔ Done when: the `_corpus/` directory contains only files from public-API Python modules, plus the top-level `LICENSE`/`NOTICE`/`COMMIT`, and re-running the script updates them in place.
- [ ] [CORE] `ingest/clone_cpp_docs.py`: clone `pytorch/cppdocs` (`--depth 1`, tracks its `main`/`master` too) — this is documentation only, no C++ source, so it goes through the same doc pipeline as Python docs (2.1's `chunk_docs.py`), just tagged `api_lang='cpp'`.
  ✔ Done when: the cppdocs clone is present and its own LICENSE/NOTICE is retained.
- [ ] [CORE] `ingest/chunk_code.py`: structure-aware **Python** code chunking using the `ast` module — one chunk per function/class, with metadata: `file_path`, `start_line`, `end_line`, `symbol_name`, docstring. These chunks exist purely so we can cite/quote exact signatures and docstrings — never to synthesize new code from.
  ✔ Done when: running on `torch/nn/modules/linear.py` produces separate chunks for `Linear`, `Bilinear`, etc., with correct line ranges.
- [ ] [CORE] `ingest/chunk_docs.py`: chunk the rst/markdown doc files (both the Python docs and the `cppdocs` C++ API pages) by heading, same metadata schema plus `api_lang: python|cpp`. Emit each chunk as an **OKF-style unit**: YAML frontmatter (`file_path`, `start_line`, `end_line`, `symbol_name`, `kind`, `api_lang`) over a markdown body, before it's split into DB columns — this is the one point in the pipeline where an intermediate on-disk artifact is worth having, since it's a human/agent-readable knowledge snapshot of the docs corpus, not just a DB-loading step.
  ✔ Done when: a sample of 5 Python-doc files and 5 cppdocs files chunks sensibly under manual review, and the OKF units are valid (frontmatter parses, required keys present).

### 2.2 Indexing in Neon
- [ ] [CORE] Table schema: `chunks(id, tsv tsvector, embedding vector NULL, file_path, start_line, end_line, symbol_name, signature, kind, api_lang)` — **no raw content column**. `embedding` is only ever populated for `kind='doc'` rows (Python docs and cppdocs alike); `kind='code'` rows always have it `NULL`. The tsvector is computed at index time (from content that is read but not stored) over `symbol_name` + `signature` + docstring/heading text. GIN index on tsv; trigram (`pg_trgm`) index on `symbol_name`; HNSW index on `embedding` (partial index, `WHERE kind = 'doc'`).
  ✔ Done when: a migration runs clean; `select * from chunks limit 1` contains no code — only text-search/vector metadata; `code` rows have `embedding IS NULL`.
- [ ] [CORE] `index/embed.py`: compute embeddings for `kind='doc'` chunks only, in batches (resilient to mid-run failure — checkpointing), and insert/update in Neon. Code chunks never go through this.
  ✔ Done when: every doc chunk has a non-null embedding, every code chunk doesn't; re-running doesn't duplicate rows.

### 2.3 Retrieval: docs first, code as a deliberate fallback
**Ordering policy:** every question retrieves from `kind='doc'` first. Falling back to `kind='code'` is a deliberate escalation, not a parallel default — it happens only when 3.1's self-grading step decides the docs don't cover what's needed (typically: the question is about an *internal mechanism* of an already-in-scope public module — e.g. "I wrote a custom LR scheduler like X; how does the internal step-count update work?" — where the public docs describe usage but not the implementation detail). This never reopens the internals-are-out-of-scope decision: it's still only the source of an already-in-scope public Python module (e.g. `torch/optim/lr_scheduler.py` itself), never `aten`/`c10`/CUDA.
- [ ] [CORE] `index/retrieve.py`: function `retrieve(query, k=8, kind=None)`. For `kind='code'` (or unspecified, symbol-shaped queries): `tsvector` full-text + `pg_trgm` fuzzy matching on symbol names only — no embedding step. For `kind='doc'` (Python docs and cppdocs): hybrid — dense (pgvector cosine) + `tsvector`, merged with RRF (start at roughly the industry-typical 70/30 vector/keyword weighting, tune from eval data). Returns **pointers** (path + line range), not content.
  ✔ Done when: searching `scaled_dot_product_attention` returns the pointer to the real definition as the top code result (including with a typo, via trigram); a semantically-phrased doc question (e.g. "how do I stop my model from overfitting") surfaces the regularization/weight-decay doc page even without matching keywords.
- [ ] [CORE] `agent/query_terms.py`: small LLM call that turns a fuzzy natural-language question into 1-3 concrete search terms (candidate symbol names / keywords) to feed `retrieve` for the code path. This is the "semantic understanding" step for code — done by the model, not by a vector index.
  ✔ Done when: "how do I randomly drop out some activations during training" resolves to search terms that include `dropout`/`Dropout` and `retrieve` finds `nn.Dropout`.
- [ ] [CORE] `index/hydrate.py`: read the actual lines from the tracked clone (or cppdocs clone) based on the pointers, ready for prompt injection. Each hydrated result carries a fixed `license` string alongside the content (the torch or cppdocs attribution, depending on source) — this is metadata, not something computed per-file, since each corpus shares one license.
  ✔ Done when: hydrating a retrieve result returns exactly the function/doc section plus its license string, and a test confirms the metadata matches the file content.
- [ ] [STRETCH] Cross-encoder reranker over the doc path's top-20 hybrid results, if eval data shows precision (not recall) is the bottleneck there — this is the standard "add a reranker when hybrid precision is low" move, not something to add speculatively.

### 2.4 Wiring and evaluation
- [ ] [CORE] Update `answer_question`: retrieve → hydrate → inject the snippets into the prompt with an explicit instruction "explain using only what's in the context; if you quote code, it must be an exact substring of the context — never write new code", and add `citations: list[{file_path, lines, license, api_lang}]` to the schema (the `license` field is the fixed attribution string from `hydrate`, not model-generated).
  ✔ Done when: answers include real citations that can be opened in the file, each with a license string attached.
  *Note: from this point the tracked clones (torch + cppdocs) are runtime dependencies — they go into the deploy image in M5. Neither requires actually installing/running the torch package — we only ever read source/doc text, never execute it.*
- [ ] [CORE] Dedicated metric `grounded_api_rate`: percentage of symbols in `symbols_referenced` that exist in the index. Run on the 15 M1 questions, compare before/after RAG.
  ✔ Done when: there is one table showing the improvement — also great material for the README.
- [ ] [CORE] Dedicated metric `quote_fidelity_rate`: percentage of answers where every quoted snippet is an exact substring of its cited hydrated chunk. This is the direct, automatable check on "no invented code" — cheaper and more central than the old ast-parse-based check from 1.3.
  ✔ Done when: the metric runs over the 15-question set and catches at least one deliberately-broken case in a test (a fake "quote" that doesn't match the source).
- [ ] [STRETCH] RAGAS on the question set (context precision/recall, faithfulness).

**Gate to M3:** `grounded_api_rate` and `quote_fidelity_rate` both improved significantly over M1, and the hallucinations from `hallucinations.md` are gone or reduced.

---

## M3 · The Agent (Weeks 5–6)

### 3.1 The manual loop
- [ ] [CORE] `agent/loop.py`: manual agent loop (~120 lines target): plan → retrieve(`kind='doc'`) → answer → **verify quote fidelity** (every quoted snippet must be an exact substring of its cited chunk) → if a quote fails verification, re-prompt once with the mismatch called out, asking it to quote correctly or drop the quote → final answer with citations. No code execution anywhere in this loop.
  ✔ Done when: "how do I use mixed-precision training" goes through the full path and returns an explanation with real, verified citations (no runnable code, no execution).
- [ ] [CORE] Self-grading on retrieval, with a code-fallback branch: after the doc retrieve, a short LLM call judges whether the context is sufficient. Three outcomes: (a) sufficient → answer from docs; (b) insufficient because retrieval missed relevant docs → rewrite the query and retry the doc search once; (c) insufficient because the question needs an *internal mechanism* of an already-in-scope public module that the docs don't cover → escalate to `retrieve(kind='code')` on the relevant symbol, and answer from the source with citations. This is the only path that reaches code chunks — it's never the first search.
  ✔ Done when: there's a test for each of the three outcomes, including one like "I wrote a custom LR scheduler; how does the internal step-count update work?" that demonstrates the (c) code-fallback path.

### 3.2 LiteLLM gateway
- [ ] [CORE] Route all calls through the LiteLLM proxy with config: primary provider + fallback, daily budget, and tag every call (`m3-loop`, `m3-grade`...).
  ✔ Done when: a per-request cost report appears in the LiteLLM logs.

### 3.3 LangGraph and comparison
- [ ] [CORE] Rewrite the loop as a LangGraph graph (the exact same nodes).
  ✔ Done when: both versions pass the same 15-question set with similar results.
- [ ] [CORE] `docs/loop-vs-langgraph.md`: short comparison — lines of code, ease of debugging, latency. One page, as an OKF unit (YAML frontmatter with `compared` and `date`).
- [ ] [STRETCH] Expose `retrieve` as an MCP server with FastMCP; test from an MCP client.
- [ ] [STRETCH] Long-term memory (user preferences, torch version) — defer if no time.

**Gate to M4:** a real question goes through plan→retrieve→answer→verify-quotes→cite end to end, with zero code execution anywhere in the system.

---

## M4 · Discipline (Week 7)

- [ ] [CORE] Wire up Langfuse: a trace for every run with a span per step (plan / retrieve / answer / verify-quotes).
  ✔ Done when: a failed run can be opened in the UI and you can see at which step it broke.
- [ ] [CORE] Expand the eval set to **40 questions** in `eval/questions_v1.jsonl`, each with: question, type, and a gold answer or automatic assertion (e.g. "the citation must point to the real symbol's actual definition line range").
  ✔ Done when: `eval/run_v1.py` runs all of them and prints: pass rate, `grounded_api_rate`, `quote_fidelity_rate`, average cost and latency.
- [ ] [CORE] Error taxonomy: classify every failure into one of 4 categories (fake symbol / missed retrieval / fabricated quote / wrong citation), log it in MLflow, and write `docs/error-analysis.md` (OKF unit: frontmatter with `category`, `count`, `eval_version`) with 3 conclusions and one improvement actually implemented.
  ✔ Done when: there is a measurable before/after for at least one improvement.
- [ ] [CORE] Cache in Upstash Redis: exact-match on (question, index version) for answers, and a cache for extracted search terms (2.3) per question.
  ✔ Done when: a repeated question returns from cache in <200ms, and hit-rate is measured.
- [ ] [STRETCH] Semantic cache (embedding similarity between questions, to catch near-duplicate phrasing) — only after the exact cache works, and only if eval data shows enough near-duplicate traffic to justify it.

**Gate to M5:** one complete eval report + one trace that can be shown in an interview.

---

## M5 · Shipping + Hardening (Week 8)

- [ ] [CORE] Minimal Gradio interface: question field, answer with highlighted quotes, clickable citations (each showing its `license` string beneath it), and a "quotes verified ✓" indicator.
- [ ] [CORE] Deploy on a free tier — pick one: HF Spaces (fastest), Modal, or Railway. No code-execution sandbox to provision — the deploy image only needs the tracked clones' text (torch source + cppdocs), never an installed torch package.
  ✔ Done when: a public link works from a clean browser, including a full query.
- [ ] [CORE] Basic auth: an API key per user (table in Neon), rate limit per key, and every query tagged to a key.
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
- Additional language bindings' public API docs (e.g. TorchScript/ONNX export surface, JS docs site) — same "docs only, no source" treatment as C++.
- WhatsApp/Slack as an additional frontend (same agent, channel wrapper).

## Stop rules
- Stuck for more than half a day on a CORE task? Cut its scope and document the cut — don't extend the time.
- Every milestone closes with a tagged commit (`m1-done`...) and a summary line in the README.
- Don't touch STRETCH while CORE is red. Don't add features that aren't on the list.
