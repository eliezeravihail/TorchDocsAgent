---
title: "Design: Content Pipeline, Agent Access, Session Flow, Orchestration, and Live Links"
kind: design
status: accepted
date: 2026-07-05
corpus: "pytorch.org documentation site (docs + tutorials + get-started)"
reference_product: "Ultralytics Docs 'Ask AI' widget (kapa.ai pattern)"
sandbox: none — answers are docs-grounded explanations with illustrative snippets, not executed code
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

**Product definition (the one-sentence version):** a docs-site assistant in the style of the Ultralytics "Ask AI" widget — it answers any question that the public PyTorch documentation site can answer, cites the live pages it used, and when a question goes beyond the site it does **not** pretend to know: it answers the covered part and points the user at the right place to look (the `[source]` link on the API page, a GitHub search, or an ecosystem project). The site is the knowledge boundary.

> **Scope history.** Two earlier assumptions have been **superseded**: (1) the corpus was five source-code modules from a pinned git clone — it is now the documentation *site*, and source code is not indexed at all; (2) answers had to contain *executed, verified-to-run* code, which required a Docker sandbox and a run-fix loop — the product actually needed is guidance: explanations with illustrative, docs-grounded snippets. **There is no sandbox.** Both changes simplify the system considerably.

## 0. The question types the assistant serves

The canonical examples (from the product owner) and what each demands of the system:

| Type | Example | What the answer is | What it demands |
|---|---|---|---|
| **Usage** | "How do I use SGD?" | The API's purpose, signature, a short snippet from/based on its docs page | Single-page retrieval, precise anchor |
| **Catalog** | "What LR schedulers exist?" | An enumerated list with one-liners and links | Retrieval must surface the *overview* page (e.g. `torch.optim` lists all schedulers on one page) — heading-chunking keeps such lists intact |
| **Recipe** | "How do I build a sequence network to detect cats?", "How do I generate music?" | A guided plan stitched from several tutorials/doc sections: data loading → model choice → training loop, each step cited | **Query decomposition**: one question → several retrieval queries, one per step |
| **Edge / partially covered** | "How do I run a fraud-detection model in the browser?" | What the docs do cover (e.g. export paths, ExecuTorch/ONNX pointers) + an honest "the rest is outside these docs" with referral links | The grade step must detect partial coverage and switch to answer-plus-referral instead of bluffing |

Recipe questions are the design driver: they are why the flow has a decomposition step and why answers cite *multiple* pages.

---

## 1. Content extraction — how, and how often

### 1.1 What the corpus is

The content of the public PyTorch documentation site, ingested per **site section** (seed list, configurable):

| Section | URL family | In v1? |
|---|---|---|
| API reference | `docs.pytorch.org/docs/{version}/**` | ✔ core |
| Tutorials | `docs.pytorch.org/tutorials/**` | ✔ core |
| Get-started / install matrix | `pytorch.org/get-started/**` | ✔ core |
| Blog, ecosystem, hub | `pytorch.org/blog/**`, … | ✘ later (add a section = add a seed) |

**Not in the corpus, by decision:** the PyTorch source code on GitHub. The agent never indexes or quotes implementation internals; it refers the user out (§2.2, §5.3).

### 1.2 How extraction works (the ingestion pipeline)

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

- **The snapshot is the source of truth for the index.** The DB stores no page text — only vectors, tsvectors, and pointers (`url`, `anchor`, snapshot path, hash). At query time content is re-read from the snapshot ("hydrate"). The live site is what users get linked to; the snapshot is what the model reads — §1.3 keeps the two in sync.
- **Heading-granular chunks.** A chunk is a coherent doc section (one method's docs, one tutorial step, one catalog table), never a fixed-size token window that cuts an example or a list in half.
- **Idempotent and incremental.** Chunk identity is `(url, anchor)`; pages whose `content_hash` is unchanged since the last crawl are skipped entirely — no re-chunk, no re-embed.

### 1.3 How often — a scheduled recrawl, because the site is alive

Tutorials get edited, the API reference moves on each PyTorch release, get-started matrices update. So refresh is **scheduled + triggered**:

| Trigger | What runs | Cadence |
|---|---|---|
| Scheduled recrawl | discover → fetch → hash-compare → re-chunk + re-embed **changed pages only** | weekly (cron); cheap because most pages are unchanged |
| New PyTorch release (docs version bump) | full pipeline against the new `docs/{version}` tree → new `index_version` | a few times a year |
| Chunker / embedding-model change | re-chunk / re-embed from the existing snapshot (no crawl) | during development |

Two invariants: every answer is **stamped** with the `index_version` and crawl date of the pages it cites, and **cache keys include `index_version`** (M4), so a recrawl that changed content automatically invalidates affected cached answers.

Cost note: a few thousand pages → tens of thousands of chunks → a one-time embedding cost in single dollars; weekly incremental runs re-embed only the diff, for cents.

---

## 2. What content the agent has access to

### 2.1 Level 1 — Hydrated retrieval results (the grounding context)

The primary channel. `retrieve(query, k=8)` returns pointers; `hydrate` reads the matching sections from the snapshot and injects them into the prompt. Per chunk the model sees: the full section text (prose + its code examples, real signatures as documented), the metadata header (`url`, `heading_path`, `kind`), and the instruction *"answer only from the provided context; if it's not there, say so and point to where to look"*. For recipe questions, several decomposed queries each contribute their k — the combined context is capped, not the per-question count.

### 2.2 What it knows *about* but not *of*: the source code and the outside world

The agent has **zero** indexed source code. What it has is **referral metadata**: every API-reference chunk carries the `[source]` GitHub link that the docs page itself displays. For "how does X work internally?" the honest answer is what the docs say **plus** "for the implementation, see: `<source link>`" — or a constructed GitHub code-search URL. The same pattern covers edge questions (§0): answer the covered part, refer out for the rest. Knowing *where to send the user* is a feature; pretending to know is the hallucination we are avoiding.

### 2.3 Level 0 — Parametric knowledge (untrusted)

The model's pretraining knowledge of PyTorch is a hypothesis generator only — it may shape the plan, the decomposition, and the retrieval queries, but it is never citable, and any API named in `symbols_used` must exist in the docs index (`grounded_api_rate` measures this).

### Explicitly out of reach at answer time

No live internet fetching during a session (links are *rendered* for the user, not *followed* by the agent — freshness is the recrawl's job); no free filesystem or snapshot browsing beyond what retrieval hydrates; no site sections outside the seed list; **no code execution** — snippets in answers are illustrations grounded in doc examples, and are statically checked (§3.2) but never run.

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
  ├─ 1. PLAN       one LLM call: classify the question type (usage / catalog /
  │                recipe / edge, §0) and emit retrieval queries —
  │                one for usage/catalog, several (per step) for recipe
  │
  ├─ 2. RETRIEVE   per query: hybrid search (pgvector dense + tsvector keyword,
  │                RRF merge) → pointers → hydrate sections from the snapshot
  │
  ├─ 3. GRADE      short LLM call: "is this context sufficient — fully, partially,
  │       │        or not at all?"
  │       ├─ fully     → continue
  │       ├─ not at all → rewrite the query, retrieve again (ONCE); still nothing →
  │       │              honest gap answer + referral links, done
  │       └─ partially → continue, but flag the gaps so the answer says what the
  │                      docs don't cover and refers out for it
  │
  ├─ 4. GENERATE   structured output → Answer {answer_md, symbols_used,
  │                torch_version, citations, referrals}; schema-repair retry once
  │
  ├─ 5. CHECK      static, no execution: code blocks parse (ast), imports are
  │       │        torch/stdlib, every symbol in symbols_used exists in the index,
  │       │        every citation pointer exists
  │       ├─ pass → continue
  │       └─ fail → one regeneration round with the specific check failures
  │                 injected; fails again → deliver with a visible warning
  │
  └─ 6. FINALIZE   attach live URLs (§5), write cache, close trace → answer
```

### 3.3 When is the decision made to deliver the answer?

At the **first** of these termination conditions — there is no open-ended wandering:

| # | Condition | Answer quality flag |
|---|---|---|
| 1 | Static checks pass | ✓ grounded |
| 2 | One regeneration round exhausted | delivered with a visible "unverified symbols" warning |
| 3 | Context absent after one query rewrite | honest gap statement + referral links |
| 4 | Hard budget hit: wall-clock timeout or LiteLLM cost cap | clean error |

Two principles: **the index is the arbiter, not model confidence** — a symbol either exists in the docs or the answer flags it; and **every loop is bounded** (rewrite ≤1, regeneration ≤1, schema repair ≤1) so cost and latency have a worst-case ceiling — which is what makes the flow a terminating graph rather than an agent that "thinks until it feels ready".

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

§3.2 contains **two cycles** (grade→rewrite→retrieve; check→regenerate) and **conditional edges** (question-type routing; fully/partially/not-at-all grading). That is the shape LangGraph models natively: each box becomes a node, the state object is `{question, qtype, queries, pointers, context, coverage, answer, check_failures, rewrite_count, regen_count}`, and the §3.3 termination table becomes the graph's edges to `END`.

### 4.3 The project's actual decisions

1. **LangChain: not used.** Provider abstraction is LiteLLM's job (binding decision in PLAN.md), and retrieval is ~100 lines of purpose-built SQL against Neon that no generic `Retriever` interface improves. LangChain would add a dependency layer with nothing left to abstract.
2. **LangGraph: used, but second.** M3 builds the loop twice: first a manual ~120-line Python loop, then a LangGraph graph with the same nodes; `docs/loop-vs-langgraph.md` records the measured comparison (LoC, debuggability, latency) instead of taking the framework on faith.
3. **When LangGraph becomes genuinely necessary:** checkpointed multi-turn sessions that survive a restart, human-in-the-loop steps, or parallel branches — the natural first one here is running a recipe question's decomposed retrieval queries **concurrently**. None are 8-week CORE scope; all are natural extensions of the graph version, which is why it exists.

---

## 5. Linking stored content to live, real links

### 5.1 The mechanism: the pointer *is* the live URL

Because the corpus is the site itself, this problem mostly dissolves. A citation is `{url, anchor, page_title, heading_path}` — the stored pointer is already the live link:

```
https://docs.pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
```

Anchors come for free: Sphinx generates a stable `id` per symbol and per heading, and `objects.inv` provides the authoritative symbol→(page, anchor) mapping for the entire API reference.

### 5.2 Keeping the link honest

The one real risk is **drift**: the live page changing after the crawl, so the user reads slightly different text than the model did.

| Risk | Mitigation |
|---|---|
| Page edited after crawl | Weekly recrawl with hash-diff (§1.3) keeps the window small; each citation carries its crawl date |
| API reference moves on release | Answers cite the **versioned** docs URL (`/docs/2.7/…`, matching the index), not `/stable/` — the versioned tree is immutable once published |
| Page deleted / URL restructured | Recrawl marks vanished `(url, anchor)` rows dead → excluded from retrieval; STRETCH (M5): HTTP-200 sweep per index build |
| Anchor renamed within a page | Fall back to the bare page URL — worse UX, never a broken link |

### 5.3 Referral links (the "where to look" feature)

For what the docs don't cover — implementation internals, out-of-corpus topics (in-browser inference, ecosystem tools) — the answer includes a **referral**, not a citation: the API page's own `[source]` GitHub link (captured at crawl time), a constructed GitHub code-search URL, or the docs search page. Referrals are visually distinct from citations in the UI: a citation means "the answer came from here"; a referral means "I don't have this — look here".

### 5.4 What the user sees (M5 UI)

Each answer renders citations as `page title › section — link` and referrals as `beyond these docs, see: [link]`. Answers delivered under termination conditions 2–3 (§3.3) carry their warning visibly. Clicking a citation lands on the exact section the model read — that is the whole trust story in one click.

---

## Summary of the five answers

1. **Extraction**: a crawl-and-index pipeline over the documentation site (discover via `objects.inv`/sitemap → fetch → heading-chunk → embed); refreshed by a weekly incremental recrawl (changed pages only) plus a full rebuild per PyTorch release — scheduled, because the site is alive.
2. **Access level**: the agent reads full hydrated doc sections (k=8 per query, several queries for recipe questions); it holds *referral metadata* for source code and out-of-corpus topics but never the content itself; parametric knowledge is never citable; nothing is executed.
3. **Session**: stateless per question; cache → plan (classify + decompose) → retrieve → grade (fully/partially/no) → generate → static check; released when checks pass or bounded retries are exhausted — the index is the arbiter, not model confidence.
4. **LangChain vs LangGraph**: component library vs. graph runtime with native cycles; LangChain not used (LiteLLM + custom SQL retrieval cover it), LangGraph used as the second loop implementation in M3 and required only when checkpointing / parallel decomposed retrieval / multi-turn arrive.
5. **Live links**: the stored pointer *is* the live URL+anchor (versioned docs tree, immutable per release); drift is bounded by the weekly recrawl; "where to look" referrals are a first-class, visually distinct answer element.
