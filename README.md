---
title: TorchDocs Agent
emoji: 🔥
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
short_description: Ask PyTorch anything, grounded in the docs with citations
---

# TorchDocsAgent

AI-powered chat agent for PyTorch — ask questions about the library, get code examples, and explore documentation through natural language. This is a personal project and is not official PyTorch team.

> The YAML header above configures the Hugging Face Space (SDK + entrypoint); GitHub just renders it as a table. See [docs/deploy-hf-spaces.md](docs/deploy-hf-spaces.md).

## Use it on Hugging Face Spaces

The agent runs as a live web app on Hugging Face Spaces — nothing to install:

**▶️ https://huggingface.co/spaces/eliezeravihail/torchdocs-agent**

Type a question in English and press **Ask** (or Enter):

- Answers are served instantly from content stored in the index, then the cited pages are **revalidated against the live docs** in the background — the index self-heals and the answer is corrected if the docs changed.
- Every answer lists the **exact documentation pages** it used as clickable citations, plus a link to the source license.
- Questions about implementation internals (source code) are **referred out** to GitHub / DeepWiki rather than guessed.

Try: *"How do I use torch.optim.SGD with momentum?"*, *"What LR schedulers are supported?"*, *"How do I build a CNN to classify images?"*

### Deploying your own Space

The repo **is** the Space: the YAML header above configures it, `app.py` is the entrypoint, and `requirements.txt` lists the dependencies. Every push to `main` auto-syncs to the Space via [`.github/workflows/sync-to-hf.yml`](.github/workflows/sync-to-hf.yml). Set these under the Space's **Settings → Variables and secrets**:

| secret | purpose |
|---|---|
| `NEON_URL` | Postgres connection string (holds vectors + pointers) |
| `TORCHDOCS_PROVIDER` | LLM provider, e.g. `openai-compat` (OpenRouter) |
| `OPENAI_COMPAT_BASE_URL` | e.g. `https://openrouter.ai/api/v1` |
| `OPENAI_COMPAT_API_KEY` | your OpenRouter key |
| `TORCHDOCS_OPENAI_COMPAT_MODEL` | comma-separated free model slugs (a fallback chain) |
| `GEMINI` / `GEMINI_API_KEY` | fallback provider key |

If the primary provider is unreachable or a free model is rate-limited, the app **self-heals** to the next model, then to any other provider that has a key — so one broken secret doesn't take the Space down. A push-triggered smoke test ([`.github/workflows/smoke-hf.yml`](.github/workflows/smoke-hf.yml)) asks the live Space a question after each deploy and fails if it can't answer. See [docs/deploy-hf-spaces.md](docs/deploy-hf-spaces.md) for the full walkthrough.

## Goals

- Answer natural-language questions about PyTorch APIs, concepts, and usage patterns — from "how do I use SGD?" through "what LR schedulers exist?" to "how do I build a network that detects cats?".
- Ground every answer in the official PyTorch documentation site, with clickable citations to the live pages used.
- Include illustrative code snippets drawn from the docs and tutorials (statically checked, not executed).
- When a question goes beyond the docs, say so honestly and point to where to look (source links, GitHub search) instead of guessing.
- Stay easy to run locally with minimal setup.

See [PLAN.md](PLAN.md) for the current roadmap and TODO list.

## Building the index

One command crawls the docs site and embeds everything into Neon
(embeddings run locally on CPU, so only `NEON_URL` is needed in `.env`; must
run on a machine with open internet access):

```bash
pip install -e .
python scripts/build_index.py
```

Safe to interrupt: crawling skips unchanged pages, embedding skips chunks
already in the DB, and every batch commits — re-running continues where it
stopped. `--skip-crawl` re-embeds the existing snapshot; `--libraries
core,tutorials` limits the run to part of the seed list.
