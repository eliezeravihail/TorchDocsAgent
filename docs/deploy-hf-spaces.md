---
kind: runbook
title: Deploy the TorchDocs Agent to Hugging Face Spaces
date: 2026-07-06
---

# Deploy to Hugging Face Spaces (free, always-on)

The app is a long-lived Gradio server: the embedding model loads once, so each
question answers in seconds. Content is read live from docs.pytorch.org when no
snapshot is bundled (`TORCHDOCS_LIVE_HYDRATE=1`, the default), so **the Space
needs no `_corpus/` and no crawl** — only the Neon index (pointers) and an LLM key.

## One-time setup

1. Create a Space: huggingface.co → New Space → **Gradio** SDK, CPU basic (free).
2. Point it at this repo, **or** push these files to the Space repo:
   `app.py`, `requirements.txt`, and the `agent/`, `index/`, `ingest/` packages.
   (Simplest: add the Space as a git remote and push.)
3. Space → Settings → **Variables and secrets**, add:
   - `NEON_URL` — the Neon connection string (the index)
   - `OPENAI_COMPAT_BASE_URL` = `https://openrouter.ai/api/v1`
   - `OPENAI_COMPAT_API_KEY` — your OpenRouter key
   - `TORCHDOCS_PROVIDER` = `openai-compat`
   - `TORCHDOCS_OPENAI_COMPAT_MODEL` = a comma-separated free-model chain
     (e.g. `poolside/laguna-xs-2.1:free,meta-llama/llama-3.3-70b-instruct:free`)

That's it — the Space boots, downloads bge-small once (~130 MB, then cached),
and serves a public URL.

## Notes

- **First boot** is slow (model download); subsequent questions are fast.
- **Live hydrate**: with no bundled snapshot the app fetches each cited page
  live (cached per-URL). To serve from a bundled snapshot instead, commit
  `_corpus/` and set `TORCHDOCS_LIVE_HYDRATE=0`.
- **Speed/quality**: free OpenRouter models are rate-limited and variable; the
  fallback chain smooths this. Loading ~$5 of OpenRouter/DeepInfra credit
  removes the throttling and is the single biggest UX upgrade.
- **Cost**: Neon free tier + HF CPU-basic free + free models = $0.
