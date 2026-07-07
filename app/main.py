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

import os
import threading
import time
from collections import defaultdict, deque

import gradio as gr
from dotenv import load_dotenv

from agent.loop import answer_agentic
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
    """Load the embedding model (and the guard classifier) so the first question isn't slow."""
    try:
        from index.embed import embed_query

        embed_query("warmup")
    except Exception as exc:  # noqa: BLE001 — warmup is best-effort
        print(f"[app] warmup skipped: {exc}")
    try:
        from agent.guard import warm_up

        warm_up()
    except Exception as exc:  # noqa: BLE001 — guard is fail-open; warmup is best-effort
        print(f"[app] guard warmup skipped: {exc}")


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


def respond(question: str, request: gr.Request = None) -> str:
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
        return render(answer_agentic(question))
    except Exception as exc:  # noqa: BLE001 — never crash the UI
        # the real error goes to the logs; the user gets a generic line, since
        # an exception string can leak hosts, model slugs, and config internals
        print(f"[app] answer failed: {type(exc).__name__}: {exc}", flush=True)
        return "Something went wrong answering that. Please try again in a moment."


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
