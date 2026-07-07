"""TorchDocs Agent — Gradio web app (M5).

A long-lived server: the embedding model loads once at startup, so each
question is answered in seconds (unlike the batch Actions runs). Ask in any
language; the agent translates the query, searches the PyTorch docs, and
answers with clickable citations.

Run locally:  python -m app.main   (needs NEON_URL + OpenRouter env, see .env)
Deploy:       Hugging Face Spaces (this file is the Space entrypoint).
"""

from __future__ import annotations

import os

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


def _warm_up() -> None:
    """Load the embedding model once so the first question isn't slow."""
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
    return "\n".join(parts)


def respond(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "Ask me something about PyTorch."
    try:
        return render(answer_agentic(question))
    except Exception as exc:  # noqa: BLE001 — never crash the UI
        return f"Something went wrong answering that: `{exc}`. Please try again."


def build_ui():
    import gradio as gr

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
    """Launch the UI with the deployment bind settings (shared by both entrypoints)."""
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


def main() -> None:
    _warm_up()
    serve(build_ui())


if __name__ == "__main__":
    main()
