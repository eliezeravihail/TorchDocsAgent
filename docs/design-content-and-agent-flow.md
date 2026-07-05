---
title: "Design: Content Pipeline, Agent Access, Session Flow, Orchestration, and Live Links"
kind: design
status: accepted
date: 2026-07-05
torch_version: "2.7.x"
related_milestones: [M2, M3, M5]
answers:
  - how and how often content is extracted
  - what content the agent can access
  - how a session runs and when the answer is finalized
  - when LangGraph / LangChain are needed and how they differ
  - how stored pointers map to live public links
---

# Design: Content Pipeline, Agent Access, Session Flow, Orchestration, and Live Links

This document complements [PLAN.md](../PLAN.md). PLAN.md says **what** to build and in which order; this document explains **how the system works** — the five design questions that the milestone list does not answer on its own. Decisions here are consistent with the binding decisions in PLAN.md (pinned torch 2.7.x, pointer-based storage, in-scope modules only).

---

## 1. Content extraction — how, and how often

### 1.1 What the source of truth is

There is exactly **one** source of truth: a local clone of the PyTorch repository at a **pinned release tag** (torch 2.7.x, `git clone --depth 1 --branch v2.7.x`), filtered to the in-scope directories:

```
torch/nn/            torch/optim/         torch/utils/data/
torch/nn/functional  torch/autograd/      docs/source/  (rst for in-scope modules only)
```

Everything else — the Neon database, the embeddings, the tsvectors — is a **derived index** over this clone, never a copy of it. The database stores no raw code or doc text; only vectors and metadata (`file_path`, `start_line`, `end_line`, `symbol_name`, `signature`, `kind`).

### 1.2 How extraction works (the ingestion pipeline)

Extraction is a batch pipeline with four deterministic steps (M2, tasks 2.1–2.2):

```
clone (pinned tag)
  → chunk_code.py   AST-based: one chunk per top-level function / class,
                     with exact line ranges, symbol name, signature, docstring
  → chunk_docs.py   heading-based splitting of .rst/.md files,
                     emitted as OKF units (frontmatter + markdown body) on disk
  → embed.py        batch embeddings + tsvector computation → insert into Neon
```

Key properties:

- **Structure-aware, not fixed-size.** Code is chunked with the `ast` module so a chunk is always a whole function or class — never a 512-token window that cuts a function in half. Docs are chunked at heading boundaries for the same reason.
- **Content is read, indexed, and discarded.** `embed.py` reads the file content to compute the embedding and the tsvector, then writes only the vector + pointer to the DB. At query time the content is re-read from the clone ("hydrate", §2.2).
- **Idempotent.** Chunk identity is `(file_path, symbol_name, start_line)` under a given `index_version`; re-running the pipeline upserts rather than duplicates, and mid-run failures resume from a checkpoint.

### 1.3 How often — snapshot-based, not scheduled

**There is no cron job and no continuous crawling.** The corpus is a versioned snapshot, and re-ingestion happens only on one of three explicit triggers:

| Trigger | What re-runs | Expected frequency |
|---|---|---|
| Bump of the pinned torch version (e.g. 2.7 → 2.8) | Full pipeline: clone → chunk → embed | A few times a year, deliberate decision |
| Change to the chunking logic | chunk → embed (clone unchanged) | During development of M2 |
| Change of embedding model | embed only (chunks unchanged) | Rare |

Why this is the right model and not a scheduler:

1. **The corpus itself is versioned.** PyTorch 2.7.x is immutable — the tag never changes. "Fresh" is meaningless for a pinned release; correctness for that version matters, not recency.
2. **Answers must be reproducible.** Every answer carries `torch_version`, and eval results are only comparable if the index did not silently shift underneath them.
3. **Index and runtime must move in lockstep.** The sandbox (M3) runs torch 2.7 and the citations point into the 2.7 tree. An auto-updating index would let these drift apart.

Each pipeline run writes an `index_version` (e.g. `torch-2.7.1__chunker-2__embed-voyage-3`) into the chunk rows and into a small `index_meta` table. The cache key (M4) and every eval report include it, so a version bump automatically invalidates stale cache entries and marks old eval runs as non-comparable.

### 1.4 Version bump procedure (for the future)

When 2.8 becomes the target: clone the new tag next to the old one, run the full pipeline into rows with the new `index_version`, run the M4 eval set against the new index, and only then flip the runtime default and delete the old rows. The old clone is kept until the flip is verified. This is a manual, gated operation — deliberately.

---

## 2. What content the agent has access to

The agent's access is layered. From most to least trusted:

### Level 1 — Hydrated retrieval results (the grounding context)

The primary channel. `retrieve(query, k=8)` returns pointers; `hydrate` reads the exact line ranges from the pinned clone and injects them into the prompt. What the model actually sees per chunk:

- the **full source** of the retrieved function/class (real signatures, real defaults, the actual implementation), or the full doc section — not a summary, not an embedding;
- the metadata header: `file_path`, line range, `symbol_name`, `kind` (code / doc);
- an explicit instruction: *"use only APIs that appear in the context"*.

The budget is k=8 chunks (post-RRF, optionally reranked). A whole-file or whole-module dump is never injected — chunk granularity is the access granularity.

### Level 2 — Execution feedback (M3)

The agent sees the **stdout/stderr and traceback** of its own generated code, run inside the sandbox (Docker, torch 2.7 CPU, 30s timeout, no network). This is a second, indirect form of corpus access: the real library rejecting a hallucinated API is ground truth the docs cannot always provide.

### Level 0 — The model's parametric knowledge (untrusted)

The model obviously "knows" PyTorch from pretraining. Design stance: this knowledge is treated as a **hypothesis generator, never as a citable source**. It may shape the plan and the query; any API that ends up in `symbols_used` must exist in the index (`grounded_api_rate` measures exactly this), and any citation must be a real pointer.

### Explicitly out of reach

- **No internet access** — neither the agent loop nor the sandbox. Live docs.pytorch.org is never fetched at answer time (links are *rendered* for the user, §5, not *read* by the agent).
- **No filesystem browsing.** In the MVP the agent cannot `ls`/`grep` the clone; it only receives what retrieval hydrates. A "read more lines around this pointer" tool is a possible M3 STRETCH, still restricted to the pinned clone.
- **Out-of-scope corpus**: C++/CUDA sources, compiled internals, other modules — not cloned, not indexed, not reachable.

---

## 3. The session flow — and when the answer is finalized

### 3.1 What a "session" is

In the MVP (through M5), a session is **one question → one answer, stateless**. No conversation memory persists between questions; multi-turn memory (user preferences, chosen torch version) is explicitly a STRETCH item in M3 and is deferred. What *does* persist across a session's lifetime is observability (a Langfuse trace per run, M4) and the answer cache (M4).

### 3.2 The request lifecycle

```
question
  │
  ├─ 0. cache check (M4) ── exact hit on (question, index_version)? → return cached answer, done
  │
  ├─ 1. PLAN       one LLM call: classify (explain vs build), extract target symbols,
  │                formulate the retrieval query
  │
  ├─ 2. RETRIEVE   hybrid search (pgvector dense + tsvector keyword, RRF merge) → 8 pointers
  │                → hydrate from the pinned clone
  │
  ├─ 3. GRADE      short LLM call: "is this context sufficient to answer?"
  │       ├─ yes → continue
  │       └─ no  → rewrite the query, retrieve again (ONCE) ── still insufficient →
  │                answer in degraded mode: explicit disclaimer, no fabricated citations
  │
  ├─ 4. GENERATE   structured output → CodeAnswer {code, explanation, symbols_used,
  │                torch_version, citations}; schema-repair retry once on parse failure
  │
  ├─ 5. RUN        sandbox execution (build-type questions)
  │       ├─ exit 0 → continue to 6
  │       └─ error  → inject traceback, regenerate — up to 3 fix rounds
  │                   3 failures → return last attempt, marked "code did not run" ✗
  │
  └─ 6. FINALIZE   validate citations (pointers exist in index), attach live URLs (§5),
                   eval/checks.py static checks, write cache, close trace → answer
```

### 3.3 When is the decision made to deliver the answer?

The answer is released at the **first** of these termination conditions — there is no open-ended wandering:

| # | Condition | Answer quality flag |
|---|---|---|
| 1 | Sandbox run succeeds (or question is explain-type and generation passed static checks) | ✓ "code runs" |
| 2 | 3 fix rounds exhausted | ✗ best attempt, marked as not verified |
| 3 | Context insufficient after one query rewrite | degraded: disclaimer, possibly a refusal for build-type |
| 4 | Hard budget hit: wall-clock timeout or LiteLLM cost cap for the request | error, clean message |

Two principles behind this table:

- **The run result is the arbiter, not the model's confidence.** For build questions, "done" means the sandbox exited 0. A self-assessment ("looks correct to me") never substitutes for execution.
- **Bounded loops everywhere.** Query rewrite: max 1. Fix rounds: max 3. Schema repair: max 1. Every loop has a counter in the state, so cost and latency have a worst-case ceiling. These bounds are what makes the flow a terminating graph rather than an "agent that thinks until it feels ready".

---

## 4. LangChain vs LangGraph — what, when, and why

### 4.1 The distinction

They are two different layers from the same ecosystem, often confused:

| | **LangChain** | **LangGraph** |
|---|---|---|
| What it is | A **component library**: LLM provider wrappers, prompt templates, retriever/vector-store interfaces, "chains" | An **orchestration runtime**: a state machine where nodes are steps and edges (including conditional and cyclic ones) define control flow |
| Control-flow shape | Linear pipelines / DAGs — data flows forward | Arbitrary graphs — **cycles are first-class** |
| State | Passed implicitly along the chain | An explicit, typed state object every node reads/writes |
| Extras | Integrations catalog | Checkpointing (pause/resume a run), human-in-the-loop interrupts, per-node retry policies, streaming of intermediate state |
| Analogy | A box of pipeline parts | A workflow engine |

The rule of thumb: **a fixed sequence of steps → LangChain-style chain is enough; a loop with "decide, act, observe, repeat" → that is a graph, and LangGraph is built for exactly that.**

### 4.2 What our flow needs

Look at §3.2: the flow contains **two cycles** (grade→rewrite→retrieve, and run→fix→run) and **conditional edges** (sufficient?, exit 0?, explain-vs-build routing). That is precisely the shape LangGraph models natively — each box becomes a node, the state object is `{question, plan, pointers, context, answer, traceback, fix_count, rewrite_count}`, and the termination table in §3.3 becomes the graph's edges to `END`.

### 4.3 The project's actual decisions

1. **LangChain: not used.** Its two main offerings are already covered by better-fitting choices: provider abstraction is LiteLLM's job (a binding decision from PLAN.md — routing, fallback, budgets), and retrieval is ~100 lines of purpose-built SQL against Neon (hybrid RRF + hydrate) that no generic `Retriever` interface improves. Adding LangChain would add a dependency layer with nothing left to abstract.
2. **LangGraph: used, but second.** M3 deliberately builds the loop **twice**: first as a manual ~150-line Python loop (`agent/loop.py`), then as a LangGraph graph with the *same* nodes. The manual loop proves we understand the control flow; the LangGraph version buys checkpointing, per-node tracing granularity, and a resumable run — and `docs/loop-vs-langgraph.md` records the measured comparison (LoC, debuggability, latency) instead of taking the framework on faith.
3. **When LangGraph becomes genuinely necessary** (vs. nice-to-have): the moment any of these arrive — multi-turn sessions that must survive a process restart (checkpointer), human-in-the-loop approval before running code, or parallel branches (e.g. retrieve code and docs concurrently). None are in the 8-week CORE scope; all are natural extensions of the graph version, which is why it exists.

---

## 5. Linking stored content to live, real links

### 5.1 The problem

The DB stores pointers into a **local clone** (`torch/nn/modules/linear.py`, lines 44–120). The user needs links that open in a browser and show the same content. The mapping must be exact and rot-proof.

### 5.2 The mechanism: URLs are computed at render time, never stored

A citation is `{file_path, start_line, end_line, symbol_name, kind}`. At **finalize** (§3.2 step 6), the app layer derives URLs from the pointer — nothing about URLs lives in the DB, so a URL-format change never requires re-indexing:

**Source link (always available, kind = code or doc):** a GitHub permalink to the pinned tag:

```
https://github.com/pytorch/pytorch/blob/v{TORCH_TAG}/{file_path}#L{start_line}-L{end_line}
e.g. https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/modules/linear.py#L44-L120
```

Because the tag is immutable, this link shows **byte-for-byte the same lines** the agent read. This is the citation's "proof" link and it cannot drift.

**Rendered-docs link (best effort, when the chunk maps to a documented symbol):** the versioned docs site follows a predictable scheme:

```
https://docs.pytorch.org/docs/{major.minor}/generated/{qualified_symbol}.html
e.g. https://docs.pytorch.org/docs/2.7/generated/torch.nn.Linear.html
```

The qualified symbol is derived from `file_path` + `symbol_name` via the module path (`torch/nn/modules/linear.py` + `Linear` → `torch.nn.Linear`, honoring the re-export convention of the in-scope packages). Chunks that have no public docs page (private helpers, doc sub-sections) simply get only the GitHub link — the docs link is an enhancement, not a requirement.

### 5.3 Why the links stay correct

| Risk | Mitigation |
|---|---|
| File moves/renames between versions | Impossible within a pinned tag; version bumps rebuild all pointers (§1.4) |
| Line numbers drift | Same — lines are exact for the pinned tag, and `hydrate` tests assert pointer↔content agreement (M2 task 2.3) |
| docs.pytorch.org restructures URLs | Only the render-time formatter changes; DB untouched. Old *versioned* doc URLs have historically remained stable |
| Dead docs link for a symbol | Fallback: emit only the GitHub permalink; STRETCH (M5): a one-time HTTP 200 sweep over generated docs URLs per index build, caching the valid set |

### 5.4 What the user sees (M5 UI)

Each answer renders its citations as: `torch.nn.Linear — source (L44–L120) · docs`, where *source* is the permalink and *docs* the rendered page. The "code runs ✓/✗" indicator from §3.3 sits next to them. Clicking *source* lands on exactly the lines that were injected into the prompt — that is the whole trust story of the project in one click.

---

## Summary of the five answers

1. **Extraction**: batch pipeline (clone → AST/heading chunking → embed) over a pinned torch 2.7.x tag; re-run only on version bump / chunker change / embedding change — snapshot-versioned, not scheduled.
2. **Access level**: the agent sees full source of retrieved functions/classes and doc sections (k=8 hydrated chunks) plus its own sandbox tracebacks; no internet, no free filesystem browsing, no out-of-scope modules; parametric knowledge is never citable.
3. **Session**: stateless per question; fixed lifecycle cache → plan → retrieve → grade → generate → run → fix; the answer is released on sandbox success, or on exhausting bounded retries (3 fixes / 1 rewrite), or on budget — execution, not model confidence, decides.
4. **LangChain vs LangGraph**: LangChain is a component library for linear pipelines — not used (LiteLLM + custom retrieval cover it). LangGraph is a graph runtime with native cycles — used in M3 as the second implementation of the loop, and becomes necessary when checkpointed sessions, human-in-the-loop, or parallel branches arrive.
5. **Live links**: DB stores pointers only; URLs are derived at render time — an immutable GitHub tag permalink (exact lines, cannot rot) plus a best-effort versioned docs.pytorch.org page per public symbol.
