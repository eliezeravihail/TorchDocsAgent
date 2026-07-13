---
title: The app/ package
kind: reference
package: app
---

# `app/` — the Gradio web app

The long-lived Gradio server that fronts the agent: the Hugging Face Space
entrypoint that loads the embedding model once, streams a live reasoning trace
while it works, and delegates every answer to `agent/`.

## Why this package exists / its boundary

`app/` owns **UI and serving concerns only**. It paints the chat surface, keeps
the server up, manages concurrency and rate limiting, streams progress, and
formats the final markdown. It does **not** answer questions — the moment a
question needs answering, `_pipeline()` hands off to `agent/` (`agent.guard`,
`agent.route.answer_routed`) and `index/` (`index.embed`, `index.freshness`).

The split is deliberate: the batch GitHub Actions runs and the eval harness call
the same `agent/` code with no Gradio in sight, and the app stays a thin,
replaceable shell. If you are looking for retrieval, tool-calling, or JSON
validation logic, it is not here — it is in `agent/`. What lives here is
everything that turns a validated `Answer` object into a responsive web page.

## The request lifecycle in the UI

`respond()` (`app/main.py`) is a **generator**. Gradio streams every value it
yields to the same output `Markdown` component, so the screen updates in place
with no page refresh. The sequence per question:

1. **Thinking note** — yields the static `THINKING_NOTE` ("🔎 Searching the
   PyTorch docs…") immediately, so the first paint happens the instant Enter is
   pressed.
2. **Live grey trace + spinner** — the real work (`_pipeline`) runs on a daemon
   **worker thread**. While it is alive, the generator loops every
   `THINKING_TICK` (0.6 s), reads the trace lines the pipeline has appended, and
   yields them via `_render_trace()`: each step in the theme's subdued grey with
   a turning Braille wheel (`THINKING_SPINNER`) on a trailing line. A multi-
   second wait reads as visible work rather than a freeze.
3. **Black answer** — when the worker finishes, the generator yields the final
   markdown (`render(answer)`), replacing the whole grey trace with the answer in
   normal text.
4. **Freshness spinner** — if the answer has citations and `index.freshness` is
   enabled, a stale-while-revalidate pass starts. The answer stays fully on
   screen while a **bare** wheel (just the spinner, no words) turns underneath it
   via `run_below()` — the user reads while the app verifies the cited pages.
5. **(Maybe) regenerated answer** — a clean check just drops the spinner. If the
   cited docs **drifted**, `freshness.refresh_pages()` self-heals the stored
   copies, `answer_routed` is re-run against the fresh content, and the new
   answer swaps in with the `↻ FRESHNESS_NOTE`. A failed check or failed
   regeneration silently keeps the answer already shown.

### Why stream the reasoning trace and not answer tokens

The obvious move — stream answer tokens like a chatbot — does not fit this
pipeline. The answer is **not free prose**; it is a validated JSON `Answer`
object assembled over several tool calls (search → read → generate → static
check). There are no partial tokens to emit until the whole thing is built and
validated. So instead the app streams the **reasoning**: the pipeline emits a
short trace line per step (which docs it searched, what it found, when it starts
writing) through the `progress` sink, and `respond()` renders those live. The
grey trace is honest visible progress for a request whose payload can only arrive
all at once.

## Serving & robustness

| Concern | Mechanism (in `app/main.py`) | Why |
| --- | --- | --- |
| **Warmup** | `_warm_up()` calls `index.embed.embed_query("warmup")` once at startup | Loads bge-small (~130 MB) before the first request, so no user eats the cold-start; also warms the guard's topicality embed. Best-effort — a warmup failure is logged, not fatal. |
| **Concurrency** | `demo.queue(default_concurrency_limit=CONCURRENCY)` | Gradio defaults every event to serial `concurrency_limit=1`; opening the queue lets many I/O-bound requests (LLM + Neon) overlap so nobody waits in line. `max_threads` is lifted in step (`max(40, CONCURRENCY*2)`) so the thread pool — which also holds threads parked in 429 back-off — never becomes the hidden ceiling. |
| **Backpressure** | `max_size=QUEUE_SIZE` | Bounds how many requests may wait behind the workers. Under a flood, extra callers get "queue full" instead of an unbounded, forever-growing line. |
| **Per-client rate limit** | `_rate_limited()` — sliding window keyed on client IP | At most `RATE_LIMIT` questions per `RATE_WINDOW` seconds per IP, so one over-eager caller can't occupy every worker slot and burn the shared free-tier LLM quota. Set `RATE_LIMIT=0` to disable. The bucket table self-prunes past 4096 entries. |
| **Fail-open errors** | `try/except` in `_pipeline` and in the worker `work()` | The UI must never crash or hang. Any exception is logged with type + message and the user gets the generic `ERROR_NOTE`; the real error never reaches the browser, since an exception string can leak hosts, model slugs, and config internals. |
| **Smoke-test contract** | `ERROR_NOTE` contains the literal phrase **"went wrong"** | The post-deploy smoke test (`scripts/smoke_space.py`) greps for that marker to detect a broken Space. Keep the phrase. The `respond` event is also registered with `api_name="respond"` so the smoke test has a stable `client.predict(..., api_name="/respond")` endpoint; because `gradio_client.predict` returns the *last* yielded value, the generator's final yield is always a real answer. |

## UX decisions & rationale

- **Send-on-Enter requires `lines=1`.** The question `Textbox` is `lines=1` (not
  2), and this is load-bearing: Gradio only fires `Textbox.submit` on a **bare
  Enter for a single-line box**. A multi-line box (`lines>1`) treats Enter as a
  newline and submits on Shift+Enter instead. `max_lines=6` still lets a long
  question grow visually — the submit rule keys off the `lines` prop, not the
  rendered height — so Enter keeps sending. Do not bump `lines` back to 2.
- **Theme-aware subdued grey that survives the sanitiser.** Trace lines use the
  inline style `color:var(--body-text-color-subdued)` (`TRACE_STYLE`), a Gradio
  CSS variable that adapts to light/dark theme. The inline style survives
  Gradio's markdown sanitiser (verified on gradio 6.20), which would strip a
  `<style>` block or class. Trace text is `html.escape`d because a step can echo
  the user's untrusted query.
- **English-only copy.** All UI strings (`INTRO`, `EXAMPLES`, notes) are English,
  matching the agent's English-only answering contract — no localisation layer.
- **In-place freshness swap.** The freshness pass never blanks the screen or
  refreshes the page: it streams updates to the same output component, keeping
  the answer visible with only a spinner line changing underneath. Verification
  is invisible unless it actually changes the answer.

## Tool & library choices

- **gradio** — provides the whole web surface (`Blocks`, `Textbox`, `Markdown`,
  `Examples`, `Button`) and, crucially, the **queue** model that makes the app
  concurrent and gives it backpressure. It is also the native Hugging Face Spaces
  SDK, so the same file is the local dev server and the deployed entrypoint. Its
  generator-streaming support is what lets `respond()` push a live trace.
- **threading** (stdlib) — the answer pipeline runs on a daemon worker thread so
  `respond()` can stay responsive and stream the spinner/trace instead of
  blocking. A `threading.Lock` guards the shared `trace` list (worker appends,
  generator reads) and the rate-limit buckets (`_RATE_LOCK`). `run_below()`
  reuses the same pattern for the freshness and regeneration passes.
- **python-dotenv** — `load_dotenv()` reads local `.env` so `python -m app.main`
  works with `NEON_URL` + OpenRouter config without exporting vars by hand; on
  the Space the real environment already carries them, so this is a no-op there.

## File by file

- **`app/main.py`** — the entire app: config constants and the `TORCHDOCS_*`
  env-var reads; the rate limiter (`_rate_limited`); warmup (`_warm_up`); the
  pure answer pipeline (`_pipeline`, no UI concerns) and markdown assembly
  (`render`, `_render_trace`); the streaming UI generator (`respond`) with its
  freshness pass; and the wiring (`build_ui`, `serve`, `main`). Running it as
  `python -m app.main` starts a local server.
- **`app/__init__.py`** — empty package marker (present but zero-length); the
  module is imported as `app.main`.
- **`app.py`** (repo root) — the Hugging Face Spaces entrypoint shim. It imports
  `_warm_up`, `build_ui`, and `serve` from `app.main`, warms the model, and
  builds the `demo` at import time (Spaces looks for a module-level `demo`), then
  calls `serve(demo)` under `__main__` for local runs. It exists so the Space has
  a top-level `app.py` while the real code stays in the `app/` package — the two
  share `serve()` so bind settings never drift between entrypoints.

## Configuration

All read once at import in `app/main.py`; override per deploy in the Space's
Variables and secrets.

| Env var | Default | What it controls |
| --- | --- | --- |
| `TORCHDOCS_CONCURRENCY` | `16` | How many questions are answered at once (Gradio `default_concurrency_limit`, and the floor for `max_threads`). Generous because requests are almost all I/O wait. |
| `TORCHDOCS_QUEUE_SIZE` | `64` | Queue `max_size`: how many requests may wait behind the workers before new ones are rejected with "queue full". |
| `TORCHDOCS_RATE_LIMIT` | `8` | Max questions per client IP per window. `0` disables the throttle. |
| `TORCHDOCS_RATE_WINDOW_SECONDS` | `60` | Length of the sliding rate-limit window, in seconds. |
| `PORT` | `7860` | Server bind port (`server_name` is fixed to `0.0.0.0`). |

Answering-side config (`NEON_URL`, the OpenRouter/LLM provider vars,
`TORCHDOCS_LIVE_HYDRATE`) is consumed by `agent/` and `index/`, not by this
package — see the deploy runbook.

## Related docs

- [`../deploy-hf-spaces.md`](../deploy-hf-spaces.md) — how the Space is created,
  configured, and smoke-tested.
- [`../design-content-and-agent-flow.md`](../design-content-and-agent-flow.md) —
  the design rationale; §5.4 covers what the user sees (citations, referrals).
- [`../agent/README.md`](../agent/README.md) — the pipeline this package
  delegates to (guard, routing, tool loop, `Answer` schema).
