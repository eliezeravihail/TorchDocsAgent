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

## Goals

- Answer natural-language questions about PyTorch APIs, concepts, and usage patterns — from "how do I use SGD?" through "what LR schedulers exist?" to "how do I build a network that detects cats?".
- Ground every answer in the official PyTorch documentation site, with clickable citations to the live pages used.
- Include illustrative code snippets drawn from the docs and tutorials (statically checked, not executed).
- When a question goes beyond the docs, say so honestly and point to where to look (source links, GitHub search) instead of guessing.
- Stay easy to run locally with minimal setup.

See [PLAN.md](PLAN.md) for the current roadmap and TODO list.

## Building the index

One command crawls the docs site and embeds everything into Neon
(requires `GEMINI_API_KEY` and `NEON_URL` in `.env`; must run on a machine
with open internet access):

```bash
pip install -e .
python scripts/build_index.py
```

Safe to interrupt: crawling skips unchanged pages, embedding skips chunks
already in the DB, and every batch commits — re-running continues where it
stopped. `--skip-crawl` re-embeds the existing snapshot; `--libraries
core,tutorials` limits the run to part of the seed list.
