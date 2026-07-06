---
kind: comparison
compared: [agent/loop.py, agent/graph.py]
date: 2026-07-06
verdict: manual loop for now; LangGraph when checkpointing / parallelism / human-in-the-loop arrives
---

# Manual loop vs. LangGraph — same agent, two control flows

Both implementations drive the **identical** three tools (`search_docs`,
`read_page`, `ask_source`), the same planner prompt, the same budgets
(6/2/1), and end in the same `answer_from_sections`. Only the control flow
differs: a Python `while` loop vs. an explicit `StateGraph`.

## Numbers

| | `agent/loop.py` | `agent/graph.py` |
|---|---|---|
| Code lines (non-comment) | **90** | 98 |
| External dependency | none (stdlib) | `langgraph` |
| State | local variables in one function | a `TypedDict` threaded through nodes |
| Control flow | `for step in range(MAX_STEPS)` + `if/elif` | nodes + a conditional edge (`_route`) |

LoC is nearly identical — the graph's node signatures and explicit state
dict cost ~8 lines over plain locals. At this size the manual loop is
slightly smaller and has zero dependency.

## Debuggability

- **Manual loop**: a stack trace points at the exact line; you can drop a
  `print`/breakpoint anywhere and read the local `transcript`. Nothing
  between you and the code. Easiest to reason about at this scale.
- **LangGraph**: the flow is *inspectable as data* — you can render the
  graph, and (with a checkpointer) replay a run node-by-node. That visibility
  is worth little for a 3-node graph but grows with complexity.

## Latency

Both make the same LLM calls (one planner call per step + one final
generation), so wall-clock is dominated by the network, not the framework.
LangGraph's per-node overhead is microseconds — immeasurable next to a
free-tier LLM call that can wait 30s on a rate limit. **No practical
difference.**

## When LangGraph earns its dependency

Nothing in *this* agent needs a graph — which is exactly why the manual loop
is the default. LangGraph becomes the right tool the moment we add any of:

- **Checkpointing** — pause a run and resume after a restart (multi-turn
  sessions, long jobs). The manual loop would have to serialize its locals
  by hand; LangGraph gives it for free.
- **Human-in-the-loop** — interrupt before an action (e.g. approve running
  code) and continue. A native graph feature; a manual bolt-on otherwise.
- **Parallel fan-out** — run several `search_docs` queries concurrently for a
  recipe question. Natural as parallel edges; awkward in a linear loop.

## Verdict

Ship the **manual loop** (`agent/loop.py`): fewer lines, no dependency,
trivially debuggable, same behavior. Keep `agent/graph.py` as the drop-in
upgrade — identical tools and prompt — for when checkpointed sessions,
approval steps, or parallel retrieval arrive. Building it twice proved the
control flow is understood, not delegated to a framework on faith.
