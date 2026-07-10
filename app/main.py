"""TorchDocs Agent — Gradio web app (M5).

A long-lived server: the embedding model loads once at startup, so each
question is answered in seconds (unlike the batch Actions runs). Ask in any
language; the agent translates the query, searches the PyTorch docs, and
answers with clickable citations.

Concurrent by default: each question is answered from request-local state
(agent/loop.py builds fresh sections/transcript/budgets per call), so many
users can be served at once. The Gradio queue is opened to TORCHDOCS_CONCURRENCY
workers instead of the framework's serial default — the work is almost all I/O
(LLM + Neon), so it overlaps cleanly and nobody waits in line.

Run locally:  python -m app.main   (needs NEON_URL + OpenRouter env, see .env)
Deploy:       Hugging Face Spaces (this file is the Space entrypoint).
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from collections import defaultdict, deque

import gradio as gr
from dotenv import load_dotenv

from agent.route import answer_routed
from agent.schemas import Answer

load_dotenv()

INTRO = (
    "# 🔥 TorchDocs Agent\n"
    "Ask anything about PyTorch — in any language. Answers are grounded in the "
    "official documentation with clickable citations; source-code questions are "
    "referred to GitHub / DeepWiki."
)

EXAMPLES = [
    "How do I use torch.optim.SGD with momentum?",
    "איזה סקדולרים נתמכים בטורץ'?",
    "How do I build a CNN to classify images, end to end?",
    "How is conv2d implemented under the hood?",
]

# Shown under the citations: a link to the PyTorch license, its name as the text.
LICENSE_NOTE = "<sub>[BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE)</sub>"

# How many questions to answer at once. The default is generous because a
# request spends nearly all its wall-clock waiting on the LLM and Neon, not on
# CPU — overlapping them is what turns "wait your turn" into "answered now".
# Override per deploy (a bigger Space, a paid LLM key) via the env var.
CONCURRENCY = int(os.environ.get("TORCHDOCS_CONCURRENCY", "16"))

# Backpressure: how many requests may WAIT behind the concurrent workers before
# new ones are turned away. Without a cap the queue grows without bound under a
# flood, and everyone in it waits forever instead of being told "busy".
QUEUE_SIZE = int(os.environ.get("TORCHDOCS_QUEUE_SIZE", "64"))

# Per-client throttle: at most RATE_LIMIT questions per RATE_WINDOW seconds per
# client IP, so one over-eager caller can't occupy every worker slot (and burn
# the shared free-tier LLM quota) by itself. 0 disables the throttle.
RATE_LIMIT = int(os.environ.get("TORCHDOCS_RATE_LIMIT", "8"))
RATE_WINDOW = float(os.environ.get("TORCHDOCS_RATE_WINDOW_SECONDS", "60"))
BUSY_NOTE = "You're asking faster than I can answer — give it a moment and try again."

# Shown the instant a question is submitted, then replaced by the answer. The
# heavy path (guard embed → retrieval → LLM) takes a few seconds, so immediate
# feedback is the difference between "is it broken?" and "it's working".
THINKING_NOTE = "🔎 Searching the PyTorch docs and drafting an answer…"
# The answer is now a fast retrieval + a ~5-8s LLM call (hydration used to be
# the long pole; it is served from the DB now). We cannot cheaply stream the
# tokens — the answer is a validated JSON object, not free prose — so instead
# the wait is ANIMATED: a spinner that ticks every THINKING_TICK seconds and a
# stage label, so a multi-second wait reads as "working", never "frozen".
THINKING_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
THINKING_TICK = 0.6  # seconds between animation frames
# retrieval+hydration is sub-second now, so after this the time is all LLM
_DRAFTING_AFTER = 1.5
# Keep the phrase "went wrong" — the post-deploy smoke test treats it as the
# failure marker (scripts/smoke_space.py). The real exception goes to the logs;
# the user never sees hosts, model slugs, or config internals.
ERROR_NOTE = "⚠️ Something went wrong answering that. Please try again in a moment."
# Appended when the post-answer freshness pass found the cited docs drifted and
# regenerated the answer from the just-refreshed content (index/freshness.py).
FRESHNESS_NOTE = (
    "\n\n<sub>↻ The docs changed since they were last indexed — this answer was "
    "regenerated from the latest content.</sub>"
)

_RATE_LOCK = threading.Lock()
_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(client_id: str) -> bool:
    """Sliding window: True if this client already used its RATE_LIMIT slots."""
    now = time.monotonic()
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS[client_id]
        while bucket and now - bucket[0] > RATE_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            return True
        bucket.append(now)
        if len(_RATE_BUCKETS) > 4096:  # keep one-off visitors from growing the table
            for key in [k for k, b in _RATE_BUCKETS.items() if not b or now - b[-1] > RATE_WINDOW]:
                del _RATE_BUCKETS[key]
        return False


def _warm_up() -> None:
    """Load the embedding model once so the first question isn't slow.

    This also covers the guard: its topicality check embeds the (translated)
    question with the same model.
    """
    try:
        from index.embed import embed_query

        embed_query("warmup")
    except Exception as exc:  # noqa: BLE001 — warmup is best-effort
        print(f"[app] warmup skipped: {exc}")


def render(answer: Answer) -> str:
    """Answer + citations + referrals as one markdown block."""
    parts = [answer.answer_md]
    if answer.citations:
        parts.append("\n---\n**Sources**")
        for c in answer.citations:
            frag = f"#{c.anchor}" if c.anchor else ""
            label = c.title or c.url
            parts.append(f"- [{label}]({c.url}{frag})")
    if answer.referrals:
        parts.append("\n**Beyond these docs**")
        for r in answer.referrals:
            parts.append(f"- [{r.reason or r.url}]({r.url})")
    if answer.torch_version and answer.torch_version != "unknown":
        parts.append(f"\n<sub>targets PyTorch {answer.torch_version}</sub>")
    if answer.citations:  # only when we actually quoted documentation
        parts.append("\n" + LICENSE_NOTE)
    return "\n".join(parts)


def _pipeline(question: str, request: gr.Request = None, out: dict | None = None) -> str:
    """The full answer pipeline → final markdown string (no UI concerns).

    `out` (optional) receives the Answer object under "answer" when one was
    generated — respond()'s freshness pass needs the citations, and returning a
    tuple would break every caller that treats the result as markdown.
    """
    question = (question or "").strip()
    if not question:
        return "Ask me something about PyTorch."
    # gradio injects `request` for real traffic; direct calls (tests) skip it
    client = getattr(getattr(request, "client", None), "host", None)
    if client and RATE_LIMIT > 0 and _rate_limited(client):
        return BUSY_NOTE
    from agent.guard import guard

    verdict = guard(question)  # one check on the raw user input, before the pipeline
    if not verdict.ok:
        return verdict.message
    try:
        # routed: simple questions take the 1-2-call grounded path (seconds),
        # multi-source shapes get the full tool loop (see agent/route.py)
        started = time.monotonic()
        answer = answer_routed(question)
        # question→answer latency is the core UX metric — log it per request so
        # the Space logs show real p50/p95, not just the eval's sampled number
        print(f"[app] answered in {time.monotonic() - started:.1f}s", flush=True)
        if out is not None:
            out["answer"] = answer
        return render(answer)
    except Exception as exc:  # noqa: BLE001 — never crash the UI
        # the real error goes to the logs; the user gets a generic line, since
        # an exception string can leak hosts, model slugs, and config internals
        print(f"[app] answer failed: {type(exc).__name__}: {exc}", flush=True)
        return ERROR_NOTE


def respond(question: str, request: gr.Request = None):
    """UI entrypoint: show a LIVE thinking indicator, then the answer.

    A generator so Gradio streams feedback the instant a question is submitted.
    The pipeline runs on a worker thread while this generator emits an animated
    spinner + stage label every THINKING_TICK seconds — so the multi-second wait
    reads as "working", not "frozen" — then yields the finished markdown.

    After the answer is shown, a stale-while-revalidate pass re-checks the
    CITED pages against the live docs (index/freshness.py). The answer stays
    fully visible the whole time, with a bare spinner (just the wheel, no
    words) under it while the check runs — the user reads while we verify —
    and everything swaps in place (Gradio streams the same output component;
    no page refresh). Clean check → the spinner simply disappears. Drift → the
    stored copies self-heal, and the answer is regenerated from the fresh
    content and swapped in with the ↻ note (the spinner keeps turning during
    the regeneration). Any freshness failure clears the spinner and leaves the
    shown answer untouched.
    The first yield is the static THINKING_NOTE (immediate paint);
    gradio_client.predict returns the LAST yielded value, so the smoke test
    still gets a real answer.
    """
    yield THINKING_NOTE

    result: dict = {}

    def work():
        try:
            result["md"] = _pipeline(question, request, out=result)
        except Exception as exc:  # noqa: BLE001 — the UI must never hang on a crash
            print(f"[app] pipeline thread failed: {type(exc).__name__}: {exc}", flush=True)
            result["md"] = ERROR_NOTE

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    frames = itertools.cycle(THINKING_SPINNER)
    started = time.monotonic()
    while True:
        worker.join(timeout=THINKING_TICK)  # wait a tick (or finish sooner)
        if not worker.is_alive():
            break
        stage = "Searching the PyTorch docs" if (
            time.monotonic() - started < _DRAFTING_AFTER
        ) else "Drafting your answer"
        yield f"{next(frames)} {stage}…"
    md = result.get("md", ERROR_NOTE)
    yield md

    # ---- stale-while-revalidate: the answer is already on screen ----------
    answer = result.get("answer")
    if answer is None or not answer.citations:
        return
    try:
        from index import freshness

        if not freshness.enabled():
            return

        def run_below(target):
            """Run `target` on a thread; while it runs, keep the ANSWER on
            screen with a bare spinner under it — just the wheel, no words
            (in-place updates: never a blank screen, never a page refresh)."""
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            while True:
                thread.join(timeout=THINKING_TICK)
                if not thread.is_alive():
                    return
                yield f"{md}\n\n<sub>{next(frames)}</sub>"

        check: dict = {}

        def verify():
            try:
                check["drifted"] = freshness.refresh_pages([c.url for c in answer.citations])
            except Exception as exc:  # noqa: BLE001 — a thread exception must not vanish
                print(f"[app] freshness check failed: {type(exc).__name__}: {exc}", flush=True)

        yield from run_below(verify)
        if not check.get("drifted"):
            yield md  # clean (or check failed): just clear the spinner line
            return

        # the text the answer was grounded in changed → answer again from the
        # just-refreshed content and swap it in, flagged so the user knows why
        print("[app] cited docs drifted; regenerating", flush=True)
        redo: dict = {}

        def regenerate():
            try:
                redo["md"] = render(answer_routed(question)) + FRESHNESS_NOTE
            except Exception as exc:  # noqa: BLE001 — keep the original answer instead
                print(f"[app] regeneration failed: {type(exc).__name__}: {exc}", flush=True)

        yield from run_below(regenerate)
        yield redo.get("md", md)  # a failed regeneration keeps the original
    except Exception as exc:  # noqa: BLE001 — never disturb the shown answer
        print(f"[app] freshness pass skipped: {type(exc).__name__}: {exc}", flush=True)
        yield md  # clear any spinner line that was showing


def build_ui():
    with gr.Blocks(title="TorchDocs Agent") as demo:
        gr.Markdown(INTRO)
        question = gr.Textbox(
            label="Your question", placeholder="How do I use a DataLoader?", lines=2
        )
        ask = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        gr.Examples(EXAMPLES, inputs=question)
        # api_name gives the post-deploy smoke test (scripts/smoke_space.py) a
        # stable Gradio endpoint to call: client.predict(..., api_name="/respond")
        ask.click(respond, inputs=question, outputs=answer, api_name="respond")
        question.submit(respond, inputs=question, outputs=answer, api_name=False)
    return demo


def serve(demo) -> None:
    """Launch the UI with the deployment bind settings (shared by both entrypoints).

    Opening the queue is what makes the app concurrent: Gradio 4/5 default every
    event to a serial `concurrency_limit=1`, so without this each user waits for
    the previous answer to finish. `max_threads` is lifted in step so the worker
    pool — which also holds threads parked in retry back-off on a 429 — never
    becomes the hidden ceiling below CONCURRENCY. `max_size` bounds how many
    requests may wait behind them, so a flood gets "queue full" instead of an
    ever-growing line.
    """
    demo.queue(default_concurrency_limit=CONCURRENCY, max_size=QUEUE_SIZE)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        max_threads=max(40, CONCURRENCY * 2),
    )


def main() -> None:
    _warm_up()
    serve(build_ui())


if __name__ == "__main__":
    main()
