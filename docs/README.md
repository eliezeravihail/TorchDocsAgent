---
title: "TorchDocsAgent — documentation index"
kind: index
---

# TorchDocsAgent — documentation

This folder is the reasoning behind the code: **why** each part exists, the
logic it follows, and the tool choices it rests on. It has two layers.

- **Cross-cutting design docs** explain the system as a whole — the product
  shape, the content pipeline, the agent contract, deployment.
- **Per-package references** (`docs/<package>/README.md`) mirror the code tree
  one-to-one: each documents a single top-level package — its boundary, its
  flow, the decisions and their rejected alternatives, and a file-by-file map.

Start with the design docs for the shape of the whole; drop into a package
reference when you're about to touch that package.

## The system in one picture

```
                         the docs.pytorch.org site
                                   │
   ingest/   discover → crawl → chunk           builds the _corpus/ snapshot
                                   │
   index/    embed → Neon (pgvector + tsvector)  storage + hybrid retrieval
                                   │  pointers + hydrated sections
   agent/    guard → route → grounded | loop → llm → Answer   the answering brain
                                   │
   app/      Gradio Space: stream the reasoning, render the answer + citations
                                   │
                              the user

   scripts/  the CLI + CI surface that builds, evaluates, and operates all of the above
   eval/     the measurement layer that keeps every layer honest
```

Data flows down the middle; `scripts/` and `eval/` wrap the column as tooling.
The knowledge boundary is firm: the docs **site** is the corpus, source code is
never indexed (it is referred out), and every answer is grounded in retrieved
sections with clickable citations.

## Per-package references

| Package | What it is | Reference |
|---|---|---|
| **`ingest/`** | The crawl → snapshot → chunk pipeline that turns the live docs site into a heading-chunked corpus. Touches no DB. | [docs/ingest/](ingest/README.md) |
| **`index/`** | The Neon/pgvector storage + retrieval layer: embeddings, hybrid search, DB-served hydration, and the self-healing freshness pass. | [docs/index/](index/README.md) |
| **`agent/`** | The answer-generation brain: guard → route → grounded/loop → LLM dispatch → validated `Answer`. | [docs/agent/](agent/README.md) |
| **`app/`** | The Gradio web app (the Hugging Face Space): serving, concurrency, and the live grey reasoning trace. | [docs/app/](app/README.md) |
| **`eval/`** | The measurement layer: static answer checks, retrieval metrics, and LLM-as-judge — wired into CI and the live answer path. | [docs/eval/](eval/README.md) |
| **`scripts/`** | The command-line + CI surface: build the index, generate synthetic data, evaluate, calibrate, and smoke-test. | [docs/scripts/](scripts/README.md) |

## Cross-cutting design docs

| Document | What it covers |
|---|---|
| [design-content-and-agent-flow.md](design-content-and-agent-flow.md) | The architecture bible: corpus scope, the ingestion pipeline, the three agent tools, the session flow, LangChain-vs-LangGraph, and how stored pointers map to live links. Start here. |
| [retrieval-gaps-and-improvements.md](retrieval-gaps-and-improvements.md) | Known retrieval weaknesses and the improvement backlog (reranking, multilingual embedder, judge model, …). |
| [loop-vs-langgraph.md](loop-vs-langgraph.md) | The measured comparison of the manual tool loop (`agent/loop.py`) against its LangGraph twin (`agent/graph.py`). |
| [deploy-hf-spaces.md](deploy-hf-spaces.md) | The Hugging Face Spaces deployment walkthrough — secrets, the sync workflow, and the post-deploy smoke test. |

See also [PLAN.md](../PLAN.md) for the milestone roadmap and the binding
decisions that constrain execution.

## Suggested reading order

1. [design-content-and-agent-flow.md](design-content-and-agent-flow.md) — the whole shape in one read.
2. [docs/ingest/](ingest/README.md) → [docs/index/](index/README.md) — how the corpus is built and retrieved (data flows up from here).
3. [docs/agent/](agent/README.md) — how a question becomes a grounded answer.
4. [docs/app/](app/README.md) — how it is served and streamed to the user.
5. [docs/eval/](eval/README.md) and [docs/scripts/](scripts/README.md) — how it is measured and operated.

## Conventions

- Each package reference has YAML frontmatter (`title`, `kind: reference`,
  `package`), then: boundary → flow → design decisions & rationale → tool
  choices → file-by-file → related docs.
- Docs describe the **code as written**, not an idealized plan. Where a plan
  item is unbuilt or a decision was superseded, the doc says so.
- These are OKF-style knowledge documents (markdown + frontmatter), the same
  convention the corpus chunks and the other `docs/*.md` reports use.
