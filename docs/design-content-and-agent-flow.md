---
title: "Design: Content Pipeline, Agent Access, Session Flow, Orchestration, and Live Links"
kind: design
status: accepted
date: 2026-07-05
corpus: "pytorch.org documentation site (docs + tutorials + get-started)"
reference_product: "Ultralytics Docs 'Ask AI' widget (kapa.ai pattern)"
related_milestones: [M2, M3, M5]
answers:
  - how and how often content is extracted
  - what content the agent can access
  - how a session runs and when the answer is finalized
  - when LangGraph / LangChain are needed and how they differ
  - how stored pointers map to live public links
---

# Design: Content Pipeline, Agent Access, Session Flow, Orchestration, and Live Links

This document complements [PLAN.md](../PLAN.md). PLAN.md says **what** to build and in which order; this document explains **how the system works**.

**Product definition (the one-sentence version):** a docs-site assistant in the style of the Ultralytics "Ask AI" widget — it answers any question that the public PyTorch documentation site can answer, cites the live pages it used, and when a question requires source-code internals it does **not** pretend to know them: it points the user at the right place to look (the `[source]` link on the API page, or a GitHub search). The site is the knowledge boundary.

> **Scope history.** An earlier revision of the plan limited the corpus to five source-code modules (`torch/nn`, `torch/optim`, `torch/utils/data`, `torch/nn/functional`, `torch/autograd`) and indexed the *code itself* from a pinned git clone. That was a feasibility scoping choice made inside this repo's planning, not an external requirement, and it has been **superseded**: the corpus is now the documentation **site**, source code is not indexed at all, and the pointer/live-link design below reflects that.

---

## 1. Content extraction — how, and how often

### 1.1 What the corpus is

The corpus is the content of the public PyTorch documentation site, ingested per **site section** (seed list, configurable):

| Section | URL family | In v1? |
|---|---|---|
| API reference | `docs.pytorch.org/docs/{version}/**` | ✔ core |
| Tutorials | `docs.pytorch.org/tutorials/**` | ✔ core |
| Get-started / install matrix | `pytorch.org/get-started/**` | ✔ core |
| Blog, ecosystem, hub | `pytorch.org/blog/**`, … | ✘ later (add a section = add a seed) |

**Not in the corpus, by decision:** the PyTorch source code on GitHub. The agent never indexes or quotes implementation internals; for "how is this implemented" questions it refers the user out (§2.3, §5.3).

### 1.2 How extraction works (the ingestion pipeline)

Extraction is a batch crawl-and-index pipeline:

```
discover   enumerate every page: Sphinx inventory (objects.inv) for the API
           reference — it maps every documented symbol to its exact page+anchor —
           plus the sitemap / toctree for tutorials and guides
  → fetch  download rendered pages, strip nav/chrome, convert HTML → markdown;
           save each page to the on-disk snapshot `_corpus/<url-path>.md`
           (+ per-page metadata: url, title, section path, content_hash, crawl date)
  → chunk  split each page by heading; a chunk = one section, with metadata
           {url, anchor, page_title, heading_path, kind: api|tutorial|guide}.
           Code blocks inside a section stay attached to that section's chunk.
           API pages also record their `[source]` GitHub link as metadata.
  → embed  batch embeddings + tsvector → upsert into Neon, keyed by
           (url, anchor) under an index_version
```

Key properties:

- **The snapshot is the source of truth for the index.** The DB stores no page text — only vectors, tsvectors, and pointers (`url`, `anchor`, snapshot path, hash). At query time content is re-read from the snapshot ("hydrate"). The live site is what users get linked to; the snapshot is what the model reads — and §1.3 keeps the two in sync.
- **Heading-granular chunks.** A chunk is a coherent doc section (e.g. one method's docs, one tutorial step), never a fixed-size token window that cuts a code example in half.
- **Idempotent and incremental.** Chunk identity is `(url, anchor)`; pages whose `content_hash` is unchanged since the last crawl are skipped entirely — no re-chunk, no re-embed.

### 1.3 How often — a scheduled recrawl, because the site is alive

Unlike a pinned git tag, the site changes: tutorials get edited, the API reference moves on each PyTorch release, get-started matrices update. So refresh is **scheduled + triggered**:

| Trigger | What runs | Cadence |
|---|---|---|
| Scheduled recrawl | discover → fetch → hash-compare → re-chunk + re-embed **changed pages only** | weekly (cron); cheap because most pages are unchanged |
| New PyTorch release (docs version bump) | full pipeline against the new `docs/{version}` tree → new `index_version` | a few times a year |
| Chunker / embedding-model change | re-chunk / re-embed from the existing snapshot (no crawl) | during development |

Two invariants make this safe:

1. **Answers are stamped.** Every answer carries the `index_version` and the crawl date of the pages it cites, so eval runs are comparable and a stale answer is identifiable.
2. **Cache keys include `index_version`** (M4), so a recrawl that changed content automatically invalidates affected cached answers.

Cost note: the docs site is a few thousand pages → tens of thousands of chunks → a one-time embedding cost measured in single dollars, and weekly incremental runs that re-embed only the diff (typically a handful of pages) for cents.

---

## 2. What content the agent has access to

### 2.1 Level 1 — Hydrated retrieval results (the grounding context)

The primary channel. `retrieve(query, k=8)` returns pointers; `hydrate` reads the matching sections from the snapshot and injects them into the prompt. Per chunk the model sees: the full section text (prose + its code examples, real signatures as documented), the metadata header (`url`, `heading_path`, `kind`), and the instruction *"answer only from the provided context; if it's not there, say so and point to where to look"*. Budget: k=8 sections — never a whole-site or whole-page dump unless the page is small.

### 2.2 Level 2 — Execution feedback (M3)

For "write me code" questions the agent still runs its generated snippet in the sandbox (Docker, matching PyTorch version, CPU, 30s, no network) and sees the traceback. The docs corpus tells it what the API promises; the sandbox tells it whether the code it wrote actually runs.

### 2.3 What it knows *about* but not *of*: the source code

The agent has **zero** indexed source code. What it does have is **referral metadata**: every API-reference chunk carries the `[source]` GitHub link that the docs page itself displays. So for "how does X work internally?" the honest answer is a summary of what the docs say **plus** "for the implementation, see: `<source link>`" — or, when no source link exists, a constructed GitHub search URL (`github.com/search?q=repo:pytorch/pytorch+<symbol>&type=code`). Knowing *where to send the user* is a feature; pretending to know the code is the hallucination we are avoiding.

### 2.4 Level 0 — Parametric knowledge (untrusted)

The model's pretraining knowledge of PyTorch is a hypothesis generator only — it may shape the plan and the retrieval query, but it is never citable, and any API named in `symbols_used` must exist in the docs index (`grounded_api_rate` measures this).

### Explicitly out of reach at answer time

No live internet fetching during a session (links are *rendered* for the user, not *followed* by the agent — freshness is the recrawl's job, §1.3); no free filesystem or snapshot browsing beyond what retrieval hydrates; no site sections outside the seed list.

---

## 3. The session flow — and when the answer is finalized

### 3.1 What a "session" is

In the MVP (through M5), a session is **one question → one answer, stateless**. No conversation memory persists between questions; multi-turn memory is explicitly a STRETCH item in M3. What *does* persist is observability (a Langfuse trace per run, M4) and the answer cache (M4).

### 3.2 The request lifecycle

```
question
  │
  ├─ 0. cache check (M4) ── exact hit on (question, index_version)? → return cached answer, done
  │
  ├─ 1. PLAN       one LLM call: classify (explain vs build vs where-is),
  │                extract target symbols, formulate the retrieval query
  │
  ├─ 2. RETRIEVE   hybrid search (pgvector dense + tsvector keyword, RRF merge)
  │                → 8 pointers → hydrate sections from the snapshot
  │
  ├─ 3. GRADE      short LLM call: "is this context sufficient to answer?"
  │       ├─ yes → continue
  │       └─ no  → rewrite the query, retrieve again (ONCE) ── still insufficient →
  │                degraded mode: say what the docs don't cover + referral link (§2.3),
  │                never a fabricated citation
  │
  ├─ 4. GENERATE   structured output → CodeAnswer {code?, explanation, symbols_used,
  │                torch_version, citations}; schema-repair retry once on parse failure
  │
  ├─ 5. RUN        sandbox execution (build-type questions only)
  │       ├─ exit 0 → continue to 6
  │       └─ error  → inject traceback, regenerate — up to 3 fix rounds
  │                   3 failures → return last attempt, marked "code did not run" ✗
  │
  └─ 6. FINALIZE   validate citations (pointers exist in index), attach live URLs (§5),
                   static checks, write cache, close trace → answer
```

### 3.3 When is the decision made to deliver the answer?

At the **first** of these termination conditions — there is no open-ended wandering:

| # | Condition | Answer quality flag |
|---|---|---|
| 1 | Sandbox run succeeds (or explain/where-is question passed static checks) | ✓ |
| 2 | 3 fix rounds exhausted | ✗ best attempt, marked not verified |
| 3 | Context insufficient after one query rewrite | degraded: honest gap statement + referral link |
| 4 | Hard budget hit: wall-clock timeout or LiteLLM cost cap | clean error |

Two principles: **execution, not model confidence, is the arbiter** for build questions; and **every loop is bounded** (rewrite ≤1, fix ≤3, schema repair ≤1) so cost and latency have a worst-case ceiling — which is what makes the flow a terminating graph rather than an agent that "thinks until it feels ready".

---

## 4. LangChain vs LangGraph — what, when, and why

### 4.1 The distinction

| | **LangChain** | **LangGraph** |
|---|---|---|
| What it is | A **component library**: LLM provider wrappers, prompt templates, retriever interfaces, "chains" | An **orchestration runtime**: a state machine where nodes are steps and edges (incl. conditional and cyclic) define control flow |
| Control-flow shape | Linear pipelines / DAGs | Arbitrary graphs — **cycles are first-class** |
| State | Passed implicitly along the chain | An explicit, typed state object every node reads/writes |
| Extras | Integrations catalog | Checkpointing (pause/resume), human-in-the-loop interrupts, per-node retries, streaming of intermediate state |
| Analogy | A box of pipeline parts | A workflow engine |

Rule of thumb: a fixed sequence of steps → a chain is enough; a loop of "decide, act, observe, repeat" → that is a graph, and LangGraph is built for exactly that.

### 4.2 What our flow needs

§3.2 contains **two cycles** (grade→rewrite→retrieve; run→fix→run) and **conditional edges** (sufficient? exit 0? explain/build/where-is routing). That is precisely the shape LangGraph models natively: each box becomes a node, the state object is `{question, plan, pointers, context, answer, traceback, fix_count, rewrite_count}`, and the §3.3 termination table becomes the graph's edges to `END`.

### 4.3 The project's actual decisions

1. **LangChain: not used.** Provider abstraction is LiteLLM's job (binding decision in PLAN.md), and retrieval is ~100 lines of purpose-built SQL against Neon that no generic `Retriever` interface improves. LangChain would add a dependency layer with nothing left to abstract.
2. **LangGraph: used, but second.** M3 builds the loop twice: first a manual ~150-line Python loop, then a LangGraph graph with the same nodes; `docs/loop-vs-langgraph.md` records the measured comparison (LoC, debuggability, latency) instead of taking the framework on faith.
3. **When LangGraph becomes genuinely necessary:** checkpointed multi-turn sessions that survive a restart, human-in-the-loop approval before running code, or parallel branches (retrieve API docs and tutorials concurrently). None are 8-week CORE scope; all are natural extensions of the graph version, which is why it exists.

---

## 5. Linking stored content to live, real links

### 5.1 The mechanism: the pointer *is* the live URL

Because the corpus is the site itself, this problem mostly dissolves. A citation is `{url, anchor, page_title, heading_path}` — the stored pointer is already the live link:

```
https://docs.pytorch.org/docs/2.7/generated/torch.nn.Linear.html#torch.nn.Linear
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
```

Anchors come for free: Sphinx generates a stable `id` per symbol and per heading, and `objects.inv` provides the authoritative symbol→(page, anchor) mapping for the entire API reference.

### 5.2 Keeping the link honest

The one real risk is **drift**: the live page changing after the crawl, so the user reads slightly different text than the model did. Mitigations, in order of importance:

| Risk | Mitigation |
|---|---|
| Page edited after crawl | Weekly recrawl with hash-diff (§1.3) keeps the window small; each citation carries its crawl date |
| API reference moves on release | Answers cite the **versioned** docs URL (`/docs/2.7/…`, matching the index), not `/stable/` — the versioned tree is immutable once published |
| Page deleted / URL restructured | Recrawl marks vanished `(url, anchor)` rows dead → excluded from retrieval; STRETCH (M5): HTTP-200 sweep per index build |
| Anchor renamed within a page | Fall back to the bare page URL — worse UX, never a broken link |

### 5.3 Referral links (the "where to look" feature)

For questions the docs don't answer — implementation internals, undocumented behavior — the answer includes a **referral**, not a citation: the API page's own `[source]` GitHub link (captured as chunk metadata at crawl time), or a constructed GitHub code-search URL, or the docs search page. Referrals are visually distinct from citations in the UI: a citation means "the answer came from here"; a referral means "I don't have this — look here".

### 5.4 What the user sees (M5 UI)

Each answer renders citations as `page title › section — link`, referrals as `for the implementation, see: [source]`, and the "code runs ✓/✗" indicator next to generated code. Clicking a citation lands on the exact section the model read — that is the whole trust story in one click.

---

## Summary of the five answers

1. **Extraction**: a crawl-and-index pipeline over the documentation site (discover via `objects.inv`/sitemap → fetch → heading-chunk → embed); refreshed by a weekly incremental recrawl (changed pages only) plus a full rebuild per PyTorch release — scheduled, because the site is alive.
2. **Access level**: the agent reads full hydrated doc sections (k=8) plus its own sandbox tracebacks; it holds *referral metadata* for source code (the `[source]` links) but never the code itself; parametric knowledge is never citable.
3. **Session**: stateless per question; cache → plan → retrieve → grade → generate → run → fix; released on sandbox success or on exhausting bounded retries — execution decides, not model confidence.
4. **LangChain vs LangGraph**: component library vs. graph runtime with native cycles; LangChain not used (LiteLLM + custom SQL retrieval cover it), LangGraph used as the second loop implementation in M3 and required only when checkpointing / human-in-the-loop / parallelism arrive.
5. **Live links**: the stored pointer *is* the live URL+anchor (versioned docs tree, immutable per release); drift is bounded by the weekly recrawl; "where to look" referrals (GitHub `[source]`, code search) are a first-class, visually distinct answer element.
