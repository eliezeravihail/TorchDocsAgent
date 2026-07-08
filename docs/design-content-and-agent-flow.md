---
title: "Design: Content Pipeline, Agent Tools, Session Flow, Orchestration, and Live Links"
kind: design
status: accepted
date: 2026-07-05
corpus: "pytorch.org documentation site (core docs + tutorials + get-started + torchvision/torchaudio; other domain-library doc sets as v1.1 seeds)"
architecture: "lean RAG core + tool-calling agent (search_docs / read_page / ask_source)"
reference_product: "Ultralytics Docs 'Ask AI' widget (kapa.ai pattern)"
source_code_strategy: "never indexed — DeepWiki MCP + [source]/GitHub referral links"
sandbox: none — answers are docs-grounded explanations with illustrative snippets, not executed code
related_milestones: [M2, M3, M5]
---

# Design: Content Pipeline, Agent Tools, Session Flow, Orchestration, and Live Links

This document complements [PLAN.md](../PLAN.md). PLAN.md says **what** to build and in which order; this document explains **how the system works**.

**Product definition (the one-sentence version):** a convenient conversational front-end to exactly two things — **the documentation as the site serves it today, and the code as it sits on GitHub `main` today** — in the style of the Ultralytics "Ask AI" widget: an agent with three tools that answers any question the site can answer, cites the live pages it used, and when a question reaches into source-code internals it consults DeepWiki and refers the user to the real code rather than pretending to know. The site is the knowledge boundary; everything beyond it is a referral, never a claim.

**Non-goal, decided:** answering per the *user's* installed PyTorch version. There is one truth — what the site and `main` say now — exactly like browsing the docs yourself. No version switcher, no multi-version index, ever in this product's scope.

> **Scope history.** Three earlier assumptions were **superseded** during planning: (1) the corpus was five source-code modules from a pinned git clone — it is now the documentation *site*; (2) answers had to contain executed, verified-to-run code (Docker sandbox + run-fix loop) — the product is guidance with illustrative snippets, **no sandbox**; (3) the flow was a fixed pipeline (plan → retrieve → grade → generate) — it is now a **tool-calling agent loop**, per the design research below.

## Research grounding (why this shape)

Decisions in this document trace to published experience of teams that built the same product:

- **kapa.ai's lessons from 100+ docs-assistant deployments**: heading-aware chunking, hybrid retrieval + reranking, citations on every answer, and honest refusal when the docs don't cover it. That is the quality bar this design copies.
- **chat-langchain (LangChain's own open-source docs bot)**: incremental re-indexing via record-manager/hash-diff — and, critically, they **tried indexing their source code for retrieval and dropped it**: raw code chunks retrieved worse than the prose docs. This externally validates our source-code strategy.
- **DeepWiki (Cognition)**: free AI wiki + Q&A over any public GitHub repo (including `pytorch/pytorch`), exposed via a public no-auth **MCP server** (`ask_question`, `read_wiki_structure`, `read_wiki_contents`). This is how we "know" the source without indexing it.
- **2026 agentic-retrieval research**: keyword search + an agent that reformulates and retries approaches RAG-level quality without heavy index machinery. Conclusion adopted here: **the agent loop matters more than index sophistication** — keep the index lean, let the agent iterate.

## 0. The question types the assistant serves

Representative examples (the set is open — coverage is defined by the corpus, not by a question list):

| Type | Example | What the answer is | What it exercises |
|---|---|---|---|
| **Usage** | "How do I use SGD?" | The API's purpose, signature, a short snippet from/based on its docs page | One `search_docs` call, precise anchor |
| **Catalog** | "What LR schedulers exist?" | An enumerated list with one-liners and links | Retrieval must surface the *overview* page; heading-chunking keeps such lists intact |
| **Recipe** | "How do I build a sequence network to detect cats?", "How do I generate music?" | A guided plan stitched from several tutorials/doc sections, each step cited | Several `search_docs` calls with different queries; `read_page` when a whole tutorial is the right context |
| **Source / internals** | "How is `conv2d` actually implemented?" | What the docs say + `ask_source` (DeepWiki) summary, always ending in a `[source]`/DeepWiki referral link | The source tool + the citation/referral distinction |
| **Edge / partially covered** | "How do I run a fraud-detection model in the browser?" | The covered part (export paths; ExecuTorch is a v1.1 seed) + an honest "the rest is outside these docs" with referrals | Recognizing partial coverage instead of bluffing |

---

## 1. Content extraction — how, and how often

### 1.1 What the corpus is

The content of the public PyTorch documentation site. docs.pytorch.org is an umbrella: besides the core docs it hosts the **domain-library doc sets** (torchvision, torchaudio, ExecuTorch, torchao, TorchRL, torchtune, torchrec, tensordict, XLA, torchtitan, …), each a Sphinx site with its own `objects.inv` — the same discover→crawl pipeline covers all of them; adding one is a seed-list line, not new code.

The seed list is **tiered**:

| Tier | Section | URL family |
|---|---|---|
| **v1 core** | Core API reference | `docs.pytorch.org/docs/stable/**` (always the latest release) |
| **v1 core** | Tutorials | `docs.pytorch.org/tutorials/**` |
| **v1 core** | Get-started / install matrix | `pytorch.org/get-started/**` |
| **v1 core** | torchvision, torchaudio | `docs.pytorch.org/{vision,audio}/stable/**` |
| **v1.1 seeds** | ExecuTorch, torchao, torchtune, TorchRL, torchrec, tensordict, XLA, torchtitan | `docs.pytorch.org/{lib}/**` |
| **referral-only** | Blog, ecosystem/landscape, Hub, forums, **GitHub source, DeepWiki** | never indexed — linked to (§5.3) |

Every chunk carries a `library` field (`core`, `vision`, `audio`, …) so retrieval can filter or route per question.

### 1.2 How extraction works (the ingestion pipeline)

```
discover   enumerate every page: Sphinx inventory (objects.inv) for API references
           — it maps every documented symbol to its exact page+anchor —
           plus the sitemap / toctree for tutorials and guides
  → fetch  download rendered pages, strip nav/chrome, convert HTML → markdown;
           save each page to the on-disk snapshot `_corpus/<url-path>.md`
           (+ per-page metadata: url, title, section path, content_hash, crawl date)
  → chunk  split each page by heading; a chunk = one section, with metadata
           {url, anchor, page_title, heading_path, library, kind: api|tutorial|guide}.
           Code blocks stay attached to their section's chunk.
           API pages also record their [source] GitHub link as metadata.
  → embed  batch embeddings + tsvector → upsert into Neon, keyed by
           (url, anchor) under an index_version
```

Key properties:

- **The snapshot is the source of truth for the index.** The DB stores no page text — only vectors, tsvectors, and pointers. At query time content is re-read from the snapshot ("hydrate"): section-level for `search_docs` results, whole-page for `read_page`.
- **Heading-granular chunks** — a chunk is a coherent doc section, never a token window that cuts an example or a catalog list in half.
- **Idempotent and incremental** — chunk identity is `(url, anchor)`; pages with an unchanged `content_hash` are skipped entirely (the chat-langchain record-manager lesson).
- **Lean by design** — embeddings via a **local open model (bge-base, 768 dims) on CPU**: no API quota or cost, the whole corpus embeds in minutes in CI, and the same model embeds queries at answer time (free-tier embedding APIs proved quota-capped far below corpus size). Plus free tsvector in Postgres. No fine-tuned embedders, no graph stores: the agent loop compensates by reformulating and retrying, which research shows is the higher-leverage investment.

### 1.3 How often — a scheduled recrawl, because the site is alive

| Trigger | Watched signal | What runs | Cadence |
|---|---|---|---|
| Scheduled recrawl | `content_hash` of every rendered page | discover → fetch → hash-compare → re-chunk + re-embed **changed pages only** | weekly (cron); cheap because most pages are unchanged |
| New PyTorch release | GitHub Releases API of `pytorch/pytorch` (`ingest/watch.py`) — a new stable tag | kicks the same recrawl **immediately** instead of waiting for the weekly slot; `/docs/stable/` now serves the new release, so the hash-diff re-embeds everything that changed | checked daily; fires a few times a year |
| Chunker / embedding-model change | (manual — the only human decision left) | re-chunk / re-embed from the existing snapshot (no crawl) | during development |

**Embedding refresh is a by-product, not a decision — and there is no version management.** The policy is *always-latest*: the index tracks whatever `/docs/stable/` serves today. A release is not a project event; it's just a week with a large hash-diff (effectively a full re-embed, ~a dollar, automatic). A page whose hash changed gets its chunks re-embedded; an unchanged page costs nothing. The watched signals deliberately exclude *commits* to `pytorch/pytorch`: hundreds land daily and almost none change the rendered docs site — the release tag and the page hashes are the signals that actually correlate with corpus change.

Two invariants: every answer is **stamped** with the `index_version` and crawl date of the pages it cites, and **cache keys include `index_version`** (M4), so a recrawl that changed content automatically invalidates affected cached answers.

---

## 2. What the agent has access to: three tools

The agent does not receive a pre-built context; it **works for its context** through exactly three tools, each with a hard call budget (§3.3):

### 2.1 `search_docs(query, library?) → [{url, anchor, title, heading_path, snippet}]`

Hybrid search (pgvector dense + tsvector keyword, RRF-merged) over the docs index, returning pointers plus hydrated section text. The agent may call it **repeatedly with reformulated queries** — for a recipe question it will naturally issue one query per step (data loading, model, training). This replaces the fixed retrieve→grade→rewrite pipeline: insufficiency is handled by the agent searching again, bounded by the call budget.

### 2.2 `read_page(url) → full page markdown`

Whole-page hydrate from the snapshot, for when a section hit isn't enough — e.g. a tutorial that must be followed end-to-end, or a catalog page whose structure matters. Guardrail: pages above a size threshold return their heading outline first, and the agent picks sections.

### 2.3 `ask_source(question) → DeepWiki answer + links`

A thin client over **DeepWiki's public MCP server** (`ask_question` on `pytorch/pytorch` and the relevant domain-library repos). Used only when the question reaches below the documented surface. Contract: whatever comes back is presented as *external* knowledge — the answer must carry a referral link (the API page's `[source]` GitHub link captured at crawl time, the DeepWiki page, or a GitHub code-search URL), and `ask_source` content is never blended into text cited to the docs. Degradation: if DeepWiki is unavailable, the tool returns only the referral links — the feature degrades to "here's where to look", never blocks an answer.

### Level 0 — Parametric knowledge (untrusted)

The model's pretraining knowledge of PyTorch shapes its search strategy but is never citable: any API named in `symbols_used` must exist in the docs index (`grounded_api_rate`), and every claim in the answer traces to a tool result.

### Explicitly out of reach

No open-web fetching (the only external call is the DeepWiki MCP); no filesystem or snapshot browsing beyond the two docs tools; no code execution — snippets are illustrations, statically checked (§3.2) but never run.

---

## 3. The session flow — and when the answer is finalized

### 3.1 What a "session" is

In the MVP (through M5), a session is **one question → one answer, stateless**. Multi-turn memory is a STRETCH item in M3. What persists is observability (a Langfuse trace per run, with a span per tool call — M4) and the answer cache (M4).

### 3.2 The request lifecycle

```
question
  │
  ├─ 0. cache check (M4) ── exact hit on (question, index_version)? → return cached, done
  │
  ├─ 1. AGENT LOOP  the LLM iterates freely within budgets:
  │       search_docs  (≤6 calls)  – reformulate and re-search until coverage
  │       read_page    (≤2 calls)  – pull a whole tutorial/catalog page when needed
  │       ask_source   (≤1 call)   – only for below-the-docs questions
  │     The loop ends when the model decides it can answer — or a budget trips.
  │
  ├─ 2. GENERATE    structured output → Answer {answer_md, symbols_used, torch_version,
  │                 citations: [{url, anchor, title}], referrals: [{url, reason}]};
  │                 schema-repair retry once
  │
  ├─ 3. CHECK       static, no execution: code blocks parse (ast), imports are
  │       │         torch/stdlib, symbols_used exist in the index, citation
  │       │         pointers exist, every ask_source-derived claim has a referral
  │       ├─ pass → continue
  │       └─ fail → one regeneration round with the failures injected;
  │                 fails again → deliver with a visible warning
  │
  └─ 4. FINALIZE    attach live URLs (§5), write cache, close trace → answer
```

### 3.3 When is the decision made to deliver the answer?

At the **first** of these termination conditions:

| # | Condition | Answer quality flag |
|---|---|---|
| 1 | Agent declares coverage and static checks pass | ✓ grounded |
| 2 | Static-check regeneration exhausted (1 round) | delivered with a visible "unverified" warning |
| 3 | Tool budgets exhausted without coverage | honest gap answer: what was found, what wasn't, referral links |
| 4 | Hard budget: wall-clock timeout or LiteLLM cost cap | clean error |

Two principles: **the index is the arbiter, not model confidence** — a symbol either exists in the docs or the answer flags it; and **every loop is bounded** (per-tool call budgets, regeneration ≤1, schema repair ≤1), so cost and latency have a worst-case ceiling. The agent has freedom *inside* the budgets; the budgets are what make the flow a terminating graph.

---

## 4. LangChain vs LangGraph — what, when, and why

### 4.1 The distinction

| | **LangChain** | **LangGraph** |
|---|---|---|
| What it is | A **component library**: LLM wrappers, prompt templates, retriever interfaces, "chains" | An **orchestration runtime**: a state machine with nodes, conditional/cyclic edges, explicit typed state |
| Control-flow shape | Linear pipelines / DAGs | Arbitrary graphs — cycles are first-class |
| Extras | Integrations catalog | Checkpointing (pause/resume), human-in-the-loop, per-node retries, streaming of intermediate state |
| Analogy | A box of pipeline parts | A workflow engine |

### 4.2 What our flow needs

The heart of §3.2 is a **tool-calling loop** — the canonical agent cycle (decide → call tool → observe → repeat), plus a check→regenerate cycle. In LangGraph terms: an agent node, a tool-executor node, a conditional edge on "answer or another tool call?", budget counters in the state, and the §3.3 table as edges to `END`. This is *the* textbook LangGraph shape.

### 4.3 The project's actual decisions

1. **LangChain: not used.** Provider abstraction is LiteLLM's job; the three tools are ours (SQL + file reads + one MCP client); nothing is left for LangChain to abstract.
2. **LangGraph: used, but second.** M3 builds the loop twice: first a manual ~100-line loop (a `while` over LLM tool-calls with budget counters — tool loops are shorter than pipelines), then the LangGraph version with the same tools; `docs/loop-vs-langgraph.md` records the measured comparison.
3. **When LangGraph becomes genuinely necessary:** checkpointed multi-turn sessions, human-in-the-loop steps, or parallel tool fan-out (several `search_docs` calls concurrently). All are natural extensions of the graph version, which is why it exists.

---

## 5. Linking stored content to live, real links

### 5.1 The mechanism: the pointer *is* the live URL

A citation is `{url, anchor, page_title, heading_path}` — the stored pointer is already the live link:

```
https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
```

Anchors come for free: Sphinx generates a stable `id` per symbol and heading, and `objects.inv` provides the authoritative symbol→(page, anchor) mapping.

Citations deliberately use the **`/stable/` URLs, not version-frozen ones** — the always-latest policy: the link the user clicks is the same "latest" the index tracks, and never goes stale as releases roll.

### 5.2 Keeping the link honest

| Risk | Mitigation |
|---|---|
| Page edited after crawl | Weekly recrawl with hash-diff keeps the window small; citations carry their crawl date |
| `/stable/` flips to a new release right after our crawl | The release watcher (§1.3) kicks an immediate recrawl, shrinking the mismatch window to hours |
| Page deleted / restructured | Recrawl marks vanished `(url, anchor)` rows dead → excluded from retrieval; STRETCH: HTTP-200 sweep per index build |
| Anchor renamed within a page | Fall back to the bare page URL — worse UX, never a broken link |

### 5.3 Referral links (the "where to look" feature)

For what the docs don't cover, the answer includes a **referral**, not a citation: the API page's own `[source]` GitHub link (captured at crawl time), the DeepWiki page (`deepwiki.com/pytorch/pytorch`), a GitHub code-search URL, or the docs search page. Referrals are visually distinct from citations: a citation means "the answer came from here"; a referral means "this is beyond the docs — look here". Every `ask_source`-derived claim must carry one.

### 5.4 What the user sees (M5 UI)

Citations render as `page title › section — link`; referrals as `beyond these docs, see: [link]`; answers delivered under termination conditions 2–3 carry their warning visibly. Clicking a citation lands on the exact section the model read — the whole trust story in one click.

---

## Summary of the five design answers

1. **Extraction**: discover (`objects.inv` + sitemap) → crawl → heading-chunk → cheap embed, over the tiered docs.pytorch.org seed list; weekly incremental recrawl + full rebuild per release. Lean index by design — the agent loop, not index sophistication, carries retrieval quality.
2. **Access**: three bounded tools — `search_docs` (hybrid, repeatable), `read_page` (whole-page hydrate), `ask_source` (DeepWiki MCP, referral-mandatory) — plus never-citable parametric knowledge. No sandbox, no open web.
3. **Session**: stateless per question; cache → bounded agent loop → generate → static check → finalize; delivery on declared coverage + passing checks, or on budget exhaustion with an honest gap answer.
4. **LangChain vs LangGraph**: LangChain not used; LangGraph is the second implementation of the tool loop (the textbook LangGraph shape) and becomes necessary with checkpointing, human-in-the-loop, or parallel tool fan-out.
5. **Live links**: the stored pointer *is* the live URL+anchor (`/stable/` — always-latest by policy); drift bounded by the weekly recrawl plus a release-triggered immediate recrawl; referrals (GitHub `main` `[source]`, DeepWiki, code search) are first-class and visually distinct from citations.
