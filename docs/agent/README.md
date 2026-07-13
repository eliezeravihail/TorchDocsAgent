---
title: "The agent/ package — answer generation"
kind: reference
package: agent
---

# The `agent/` package

The answer-generation brain: it takes one vetted PyTorch question and returns a single docs-grounded `Answer` — with validated citations, honest referrals, and a bounded worst-case cost.

## Why this package exists / its boundary

`agent/` owns everything from "a question has arrived" to "a structured `Answer` is ready to render". It owns the **trust boundary check** (`guard.py`), the **cost decision** (`route.py`), the two answering strategies (`grounded.py` single-shot and `loop.py` / `graph.py` multi-tool), the **provider dispatch and structured-output contract** (`llm.py`), the **output schema** (`schemas.py`), and the **tool surface** the agent drives (`tools.py`, `tools_exec.py`).

It does **not** own retrieval or content. The three tools are thin: `search_docs` calls `index.retrieve` + `index.hydrate`, `read_page` calls `index.hydrate`, and the guard's topicality check calls `index.retrieve.top_distance`. All embedding, hybrid search (pgvector + tsvector RRF), snapshot hydration, and the index itself live in `index/`. `agent/` treats retrieval as a pointer-returning oracle and works entirely with the hydrated section dicts it hands back. The static answer checks (parses/imports/symbols) live in `eval/checks.py`; `grounded.py` merely wires them into the live path. So the boundary is: **`agent/` decides how to think about a question and how to phrase the answer; `index/` decides what the docs say.**

## The flow

A question travels through the package like this:

```
question
  │
  ├─ guard.py         one check at the trust boundary (length → language → topicality)
  │                   fail → refusal string, never reaches the LLM
  │
  ├─ route.py         zero-LLM regex heuristic: multi-source shape?
  │      ├─ yes ─────────────────────────────▶ loop.py  (answer_agentic)
  │      └─ no  ─────▶ grounded.py (answer_grounded, one pass)
  │                        └─ produced NO citations? ─▶ escalate to loop.py
  │
  ├─ loop.py / graph.py   planner ⇄ tools within budgets, accumulating sections
  │                       (search_docs ≤6, read_page ≤2, ask_source ≤1)
  │
  ├─ grounded.answer_from_sections   the shared terminus for BOTH paths
  │                       build context → llm.answer_question → static-check
  │                       repair (≤1) → validate_citations
  │
  └─ llm.py + schemas.py  provider dispatch → structured JSON → validated Answer
```

**The routing heuristic** (`route.needs_loop`) is a single compiled regex over the raw English question, matching the shapes where one retrieval pass is provably not enough: catalog (`what/which … exist / are available`, `list all …`), compare (`difference between`, `X vs Y`, `should I use X or Y`), recipe (`build/train … a model/network/classifier`, `end-to-end`, `from scratch`), and internals (`how is … implemented`, `under the hood`). Everything else goes to the single-shot grounded path. The heuristic is deliberately coarse because it decides **cost, not correctness** — it is allowed to be wrong.

**Escalation on no citations** is the safety net that makes coarseness acceptable. When `answer_routed` sends a question to `answer_grounded` and the returned `Answer` has an empty `citations` list, that means one fixed retrieval pass found nothing groundable. Rather than ship an unsourced reply, `route.py` escalates the same question to `answer_agentic`, which can reformulate and re-search. The heuristic thus fails open in both directions: a misrouted multi-source question still gets a (slower) correct loop answer, and a misrouted simple question that comes back empty gets a second, more thorough attempt.

Both the loop and the single-shot path converge on `grounded.answer_from_sections`, so the grounding contract (context-only generation, one static-check repair round, citation validation) is applied **exactly once and identically** no matter how the sections were gathered.

## Design decisions & rationale

**The grounding contract.** `GROUNDED_SYSTEM` in `grounded.py` instructs the model to answer *only* from the numbered context sections, be concise, cite every section it used verbatim, and add a `referral` instead of guessing when the context falls short. That instruction is enforced, not trusted:
- `validate_citations` drops any citation whose `(url, anchor)` (or bare `url`) is not in the sections actually provided — the model cannot cite a page it wasn't shown.
- `_regenerate_if_checks_fail` runs `eval/checks.run_checks` (code blocks parse via `ast`, imports are torch-family/stdlib only, every `symbols_used` entry appears verbatim in the prose) and, on failure, re-asks once with the failure reasons injected. The repair is kept **only if it is strictly cleaner** (fewer failing checks); it never blocks the user.
- Per-section context is capped at `SECTION_CHAR_LIMIT` (2500 chars) with a **visible** truncation marker and a log line, so the model can referral out rather than silently answer from a cut-off view.

**Route before loop (cost).** The loop costs ~5–13 LLM calls per question (planner rounds + answer + possible repair); on the free-tier deployment models that is minutes of wall-clock. The loop's value is real only for multi-source questions — measured agentic coverage 0.567 vs 0.133 single-shot — while a plain usage question is answered well by one retrieval pass plus one generation. A zero-LLM-call heuristic pays nothing to keep the common case cheap.

**The guard uses embedding distance as the on-topic policy.** Membership in the docs' embedding space *is* the policy: `_is_on_topic` embeds the question, takes the cosine distance to its nearest doc chunk, and blocks anything beyond `DEFAULT_TOPICALITY_MAX_DISTANCE` (0.35, calibrated 2026-07-07 to sit between the worst on-topic 0.305 and the best off-topic 0.371). This is deliberate: an off-topic request and a prompt injection ("ignore your rules and …") both land far from the corpus and get the same refusal, so **no dedicated injection classifier is needed** — an earlier design used one, which cost an extra model call and still missed injections wrapped in on-topic questions. What passes the gate is safe anyway because of the downstream grounding contract. The guard is **fail-open**: any retrieval error is logged and allowed, so the guard can never take the app down, and the whole guard is toggleable via `TORCHDOCS_GUARD=0`.

**English-only.** The corpus and the bge-small embedder are English-only, so `looks_english` (a cheap mostly-ASCII heuristic, tolerating a couple of stray non-Latin chars) bounces foreign input with a "please rephrase in English" refusal. The rejected alternative was a per-question translation LLM call, which dominated latency for non-English input. A multilingual embedder would remove this limit (noted in `docs/retrieval-gaps-and-improvements.md`).

**The provider fallback chain.** `llm.py` dispatches across three providers — `openai-compat` (OpenRouter/DeepInfra/Nebius/…, the free-tier deployment default), `anthropic` (the paid production path), and `gemini`. `default_provider()` honors `TORCHDOCS_PROVIDER` or auto-detects from whichever API key is set; `_provider_chain` then puts the preferred provider first and appends every *other* configured provider, so one broken or misconfigured secret can't take the deploy down — the answer path self-heals to another provider. Within `openai-compat`, `TORCHDOCS_OPENAI_COMPAT_MODEL` is a comma-separated model chain (default `tencent/hy3:free` → `meta-llama/llama-3.3-70b-instruct:free`) so a rate-limited or retired slug falls through to the next. A process-wide **cooldown circuit breaker** (`_COOLDOWNS`) records models/providers that just failed (60s on a 429, an hour on a 404) so other in-flight requests skip them instead of each paying the full retry+sleep cost. There is hard-won secret hygiene here too: `_env_secret` strips trailing newlines (a pasted key with a newline becomes an illegal `Authorization` header and looks like a total outage), and `_redact` masks Bearer/`sk-` tokens before any exception text is logged.

**The `progress` reasoning-trace sink.** `answer_grounded`, `answer_agentic`, and `answer_routed` all accept an optional `progress` callable. It is a sink for short human-readable trace lines ("🔍 searched …", "📄 found: …", "✍️ writing the answer", "↻ no sources yet — searching more thoroughly") that the web UI streams in grey while the answer assembles. `loop._humanize` turns the terse transcript records into these lines, falling back to the raw line on any shape it doesn't recognise so a format change degrades gracefully. It defaults to `None` (no-op), so scripts and tests are unaffected.

**The LangGraph twin.** `graph.py` reimplements the manual loop as a LangGraph state machine (`planner → tools → planner` cycle, `planner/tools → generate → END`). It is intentionally the *second* implementation: it shares the exact same tools, budgets, `_plan` planner, and forced seed search, and its tool step is literally `tools_exec.execute_tool` — so the two drivers **cannot drift**. Only the control flow differs (an explicit graph vs a Python `while`-loop). The point is a controlled comparison of lines-of-code, debuggability, and latency, recorded in `docs/loop-vs-langgraph.md`, and a ready path to features that genuinely need a graph runtime (checkpointed multi-turn, human-in-the-loop, parallel tool fan-out).

## Tool & library choices

| Library | Used for | Why (and what was rejected) |
|---|---|---|
| **pydantic** | `schemas.py` `Answer`/`Citation`/`Referral`; every provider validates into `Answer` | One schema drives structured output across all three SDKs (`Answer.model_json_schema()` becomes a Gemini `response_schema`, an OpenAI `json_object` prompt, and an Anthropic forced-tool `input_schema`) and the validation error text feeds the single repair round. |
| **anthropic** SDK | `_answer_anthropic` / anthropic raw path | Forced tool call (`tool_choice` → `submit_answer`) is the most reliable structured-output mechanism on the paid production model. |
| **google-genai** SDK | `_answer_gemini` | Native `response_schema` + `application/json` mime type; free tier for cheap runs. |
| **openai** SDK | `_answer_openai_compat` + shared `_compat_client`/`_compat_complete` | Talks to any OpenAI-compatible host (OpenRouter default). One pooled client per `(base_url, key, timeout)` keeps httpx keep-alive across the ~13 calls a question makes. |
| **No LangChain** | — | Provider abstraction is `llm.py`'s own dispatch layer; the three tools are ours (SQL retrieval + snapshot reads + referral links); nothing is left for LangChain to abstract, so it is not a dependency. Older design docs mention **LiteLLM and Langfuse — neither is in the real code**; `agent/llm.py` is the hand-rolled dispatch, and observability is `print`-to-logs. |
| **langgraph** | `graph.py` only | The orchestration runtime for the twin loop; imported lazily inside `build_graph()` so the manual path has no hard dependency on it. |

A dispatch layer instead of a framework is the recurring theme: the structured-output contract is uniform (schema-valid `Answer` or `GenerationError`, one repair round) but each provider reaches it its own way, and keeping that in ~640 lines of `llm.py` bought the exact cooldown/fallback/secret-hygiene behavior the free-tier deployment needed.

## File by file

- **`__init__.py`** — empty package marker; imports are done lazily at call sites throughout the package (to keep optional SDKs like `langgraph`/`google-genai` from being hard import-time dependencies).
- **`guard.py`** — the single input guardrail, run once on the raw user question at the trust boundary. Three cheapest-first checks (length → English → topicality). *Key decision:* embedding-distance membership is the entire on-topic policy — off-topic and prompt injection get the same refusal, no separate classifier — and it fails open on any error.
- **`route.py`** — the cost router. A zero-LLM regex sends multi-source shapes to the loop and everything else to the single-shot path. *Key decision:* a grounded answer with no citations escalates to the loop, which lets the heuristic be coarse without shipping unsourced replies.
- **`grounded.py`** — the single-shot path *and* the shared answer terminus (`answer_from_sections`). Builds the numbered context, generates, runs the static-check repair, and validates citations. *Key decision:* citations are validated against the provided sections, so the model can never cite a page it wasn't shown.
- **`loop.py`** — the manual agent loop. A planner LLM returns a JSON action each step; tools run within budgets (`search_docs:6, read_page:2, ask_source:1`) until the model answers or a budget trips. *Key decision:* a forced seed `search_docs` on the raw question runs *before* the first planner call, so a rate-limited planner never blocks the obvious first retrieval.
- **`graph.py`** — the LangGraph twin of `loop.py`, same tools/budgets/planner. *Key decision:* the tool node delegates to the shared `tools_exec.execute_tool`, guaranteeing the two drivers can't diverge behaviorally.
- **`tools_exec.py`** — one tool-execution step (`do_search`, `execute_tool`) shared by both drivers. Mutates the section/referral/seen/transcript accumulators in place with consistent dedup and observation strings. *Key decision:* extracting the dispatch + dedup here is exactly what keeps loop and graph identical; budget accounting stays with each driver.
- **`tools.py`** — the three tool functions themselves. `search_docs` (hybrid retrieval, optional `kind`/`library` filter, unknown `kind` degrades to unrestricted), `read_page` (whole-page hydrate; rejects a non-URL with a self-correcting error message), `ask_source` (referral-only). *Key decision:* `ask_source` never returns claims about the code — only DeepWiki + GitHub-code-search referral links — so it works with no network access and keeps source knowledge strictly out of docs-cited text.
- **`llm.py`** — the provider dispatch layer: one question in, one validated `Answer` out, across gemini/anthropic/openai-compat with a self-healing provider+model fallback chain, cooldown circuit breaker, retry/backoff, and secret redaction/stripping. *Key decision:* a hand-rolled dispatch (not LiteLLM/LangChain) because the free-tier reality — dead slugs, per-minute and per-day rate limits, newline-corrupted secrets — needed bespoke resilience.
- **`schemas.py`** — the pydantic output contract: `Answer` (`answer_md`, `symbols_used`, `torch_version`, `citations`, `referrals`) plus `Citation` and `Referral`. *Key decision:* the distinction between a `Citation` ("the answer came from here") and a `Referral` ("this is beyond the docs — look here") is encoded in the type system, mirroring the product's knowledge-boundary rule.

## Related docs

- [`../design-content-and-agent-flow.md`](../design-content-and-agent-flow.md) — the design rationale for the tools, session flow, and live-link mechanism this package implements.
- [`../loop-vs-langgraph.md`](../loop-vs-langgraph.md) — the measured comparison of the manual `loop.py` against the `graph.py` twin.
- [`../index/README.md`](../index/README.md) — the sibling package that owns retrieval, hydration, and the docs index this package delegates to.
- [`../eval/README.md`](../eval/README.md) — the evaluation harness and `eval/checks.py`, the static answer checks wired into `grounded.py`.
