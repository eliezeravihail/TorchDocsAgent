---
title: "scripts/ — operator CLI entrypoints and CI jobs"
kind: reference
package: scripts
---

# `scripts/` — the command-line surface

The operator/CLI layer: every way a human (or a GitHub Actions job) drives the
system by hand — build and refresh the index, generate synthetic index-side
content, evaluate, calibrate the guard, smoke-test, and dump inventories for
diagnostics.

## Why this package exists / its boundary

These scripts are **thin wrappers**. They own no business logic of their own —
they parse argv, `load_dotenv()`, fail fast on a missing env var, and then call
into `ingest/`, `index/`, and `agent/`. The real work lives there:

- `build_index.py` calls `ingest.discover.discover`, `ingest.crawl.crawl`, and
  `index.embed.build_index` — it just sequences them and prints timing.
- `ask.py` calls `agent.guard.guard` then `agent.loop.answer_agentic`.
- `search.py` calls `index.retrieve.retrieve` + `index.hydrate.hydrate_section`.
- `calibrate_guard.py` calls `index.retrieve.top_distance`.

The one piece of genuine logic that *does* live here is the batch-job
scaffolding in `generate_glosses.py` (`git_checkpoint`, batching, resume) —
because it is an operational concern (surviving a cancelled CI run), not a
system concern. `generate_questions.py` imports that scaffolding rather than
duplicating it.

Heavy imports are deliberately done **inside `main()`**, after the env-var
check, so a missing `NEON_URL` prints one clean line instead of a stack trace
from importing a DB module at file scope.

`__init__.py` is empty — it exists only to make `scripts` a package so
`generate_questions.py` can `from scripts.generate_glosses import ...`.

## Script catalog

| Script | What it does | Run it | Workflow(s) that invoke it |
|---|---|---|---|
| `build_index.py` | End-to-end index build: discover → crawl → chunk+embed → Neon. Every stage resumable; every embed batch commits. | `python scripts/build_index.py [--skip-crawl] [--skip-embed] [--libraries core,vision]` | **build-index.yml** (`--skip-embed` then `--skip-crawl`), **backfill-content.yml** (`--skip-crawl`) |
| `ask.py` | Answer one question end-to-end from the CLI: guard, then the full agent tool loop; prints answer + citations + referrals. | `python scripts/ask.py "how do I build a CNN?"` | — (local only) |
| `search.py` | Retrieval-only acceptance test: run hybrid retrieve, print the ranked pointers + a snippet of the top hit. No LLM. | `python scripts/search.py "scaled_dot_product_attention" [-k 8] [--library vision]` | — (local only) |
| `calibrate_guard.py` | Recompute the guard's topicality cutoff against the **live** index: distances for 100 on-topic + 100 off-topic + borderline probes, plus a suggested threshold. Prints only. | `python scripts/calibrate_guard.py` | **calibrate-guard.yml** |
| `generate_glosses.py` | Batch LLM job: a 1-sentence Contextual-Retrieval gloss per api page → `index/glosses.jsonl`. Batched, resumable, per-batch flush, optional `--push` checkpoint. | `python scripts/generate_glosses.py [--limit N] [--batch N] [--push]` | **build-index.yml**, **generate-glosses.yml** |
| `generate_questions.py` | Batch LLM job: a few QuOTE-style hypothetical questions per api page → `index/questions.jsonl`. Same scaffolding as glosses. | `python scripts/generate_questions.py [--limit N] [--batch N] [--push]` | **build-index.yml**, **generate-glosses.yml** |
| `smoke.py` | Preflight: one Neon write/read, one Gemini call, one local bge-small embedding, one optional Anthropic call. Exits non-zero on any required failure. | `python scripts/smoke.py` | — (local preflight) |
| `smoke_space.py` | Post-deploy health check against the live HF Space: poll runtime until RUNNING, ask a real question via the Gradio API, fail on an LLM/transport error marker. | `python scripts/smoke_space.py` | **sync-to-hf.yml** |
| `dump_docs_inventory.py` | Dump external ground truth: every documented symbol/page the docs SITE publishes (`objects.inv` + sitemap) → `eval/docs_inventory.jsonl`. Needs network to docs.pytorch.org. | `python scripts/dump_docs_inventory.py` | **build-index.yml** (inventory job) |
| `dump_index_manifest.py` | Dump internal ground truth: every distinct page actually in the `chunks` table → `eval/index_manifest.jsonl`. Needs `NEON_URL`. | `python scripts/dump_index_manifest.py` | **build-index.yml** (inventory job) |
| `coverage_diff.py` | Diff the two dumps: pages the site publishes but our index is missing. A non-empty gap is a pipeline bug. | `python scripts/coverage_diff.py` | **build-index.yml** (inventory job) |
| `__init__.py` | Empty package marker (enables `from scripts.generate_glosses import ...`). | — | — |

Note: the **eval** jobs (`eval.yml`) run `python -m eval.run_retrieval`,
`eval.diagnose_retrieval`, `eval.run_agentic`, `eval.run_judge` — those live in
the `eval/` package, not here. `ci.yml` runs `ruff` + `pytest`, and
`security.yml` runs Trivy; neither invokes a script in this package.

## Operational flows

### (a) Build / refresh the index

`build_index.py` is the whole pipeline. Locally you run it plain; in CI the
**build-index.yml** workflow splits it so glossing can happen *between* crawl
and embed:

1. `build_index.py --skip-embed` — crawl only, refresh the on-disk `_corpus`
   snapshot, never touch Neon (so the crawl-only path doesn't even require
   `NEON_URL`).
2. `generate_glosses.py --limit 0 --push` then `generate_questions.py --limit 0
   --push` — enrich only the pages that don't have a gloss/question set yet.
3. `build_index.py --skip-crawl` — embed the snapshot into Neon. A changed
   `glosses.jsonl`/`questions.jsonl` bumps the embed recipe, so only the touched
   pages re-embed.

Resumability is the design point: crawling skips unchanged pages, embedding
skips chunks whose hash is already in the DB, every embed batch commits. Kill it
and re-run — it continues. Both **build-index.yml** and **backfill-content.yml**
share the `concurrency: build-index` lock so two writers never race the DB.
`--libraries core,vision` restricts the run to a subset of the seed list.

### (b) Generate synthetic index-side content (glosses / questions)

`generate_glosses.py` and `generate_questions.py` are the two batch LLM jobs.
Both walk every `api`-kind page in the snapshot, batch them into LLM calls, and
append JSON lines to `index/{glosses,questions}.jsonl`, flushing after each
batch. They are **resumable**: URLs already present in the output file are
skipped, so a rate-limited death just means "run it again." Core-torch pages are
glossed first, so a partial run still covers the pages that matter most. After 5
failed batches they stop, assuming the provider is down. `generate_questions.py`
imports `api_pages`, `existing_urls_of`, and `git_checkpoint` directly from
`generate_glosses.py` — one pipeline shape, not two.

The `--push` flag turns on `git_checkpoint`: commit **and push** the jsonl after
every batch. See rationale below.

### (c) Evaluate

Retrieval acceptance from the CLI is `search.py` (pointers only, no LLM) and
end-to-end answering is `ask.py`. The scored benchmarks
(recall/MRR, agentic, judge) are the `eval/` package run via `eval.yml`, not
scripts here. `coverage_diff.py` is the eval-adjacent check that the corpus the
index holds actually matches the corpus the docs site publishes.

### (d) Calibrate the guard

`calibrate_guard.py` re-derives the topicality distance cutoff against the live
index. It runs three groups — 100 on-topic (must all pass), 100 off-topic
(should all block), and a handful of borderline/injection probes to eyeball —
through the guard's `top_distance` path and prints every distance plus a
suggested `TORCHDOCS_TOPICALITY_MAX_DISTANCE` (the midpoint between the worst
on-topic and best off-topic distance). It **only prints**; a threshold change is
a policy decision, so a human reads the log and edits the constant in
`agent/guard.py` by hand. Run it (via **calibrate-guard.yml**) after any corpus
change big enough to shift the distance distribution — a re-embed or a new doc
set.

### (e) Smoke-test — locally and post-deploy

Two different tests for two different moments:

- `smoke.py` — **before building anything**, verify each external connection
  works: Neon write/read, Gemini, local embedding, optional Anthropic. A missing
  key skips (Anthropic) or fails with a clear message rather than a traceback.
- `smoke_space.py` — **after deploy**, verify the live Space actually answers.
  It runs in **sync-to-hf.yml** right after the push (GitHub Actions can reach
  both the Space and the LLM provider; the dev sandbox can't). It polls the HF
  runtime API until RUNNING, asks a real question through the Gradio
  `/respond` endpoint, and fails the job if the answer contains an
  LLM/transport error marker — so a broken deploy is a red check, not a Space
  that silently serves errors. An empty-index answer warns but doesn't fail
  (that's a separate subsystem).

### (f) Inventory / diagnostics

Three scripts build and compare two ground-truth files, all in the inventory job
of **build-index.yml**:

- `dump_docs_inventory.py` → `eval/docs_inventory.jsonl` — what the site
  publishes (external truth; must run where docs.pytorch.org is reachable).
- `dump_index_manifest.py` → `eval/index_manifest.jsonl` — what our index
  actually holds (internal truth; needs `NEON_URL`).
- `coverage_diff.py` — pages in the first but not the second: a page the docs
  document that our system can never retrieve. Both dumps are committed so the
  diff can run offline.

## Design decisions & rationale

**Batch git-checkpoint (`git_checkpoint` + `--push`).** A gloss/question pass
over the ~3.6K-page corpus is a multi-hour LLM job. The batches are flushed to
disk, but on a GitHub runner that file only reaches the repo via the workflow's
final commit step — so a cancel or a job timeout part-way through throws away
everything generated in the run. `--push` commits and pushes the jsonl every few
batches instead, so a killed run keeps its progress. Every git failure (unset
identity, a push race with a concurrent enrichment run, a rebase conflict) is
logged and swallowed — a missed checkpoint just defers to the final commit; it
must **never** kill the long run. The committer identity is injected per-command
(`git -c user.name=...`) so no global config or extra workflow step is needed,
and the commit message carries `[skip ci]` so a checkpoint push doesn't kick off
a CI run. It is opt-in (`--push`) so local runs never commit.

**`--push` off by default.** Local runs of the batch jobs should produce a
jsonl and nothing else; only CI, which needs to persist progress across a
possible cancellation, turns pushing on.

**Smoke tests exist to fail fast on deploy.** `smoke.py` stops you before an
hour of crawling if a credential is wrong; `smoke_space.py` turns a broken
deploy into a red check on the very run that produced it (one workflow does both
push and verify). Both do exactly one real round-trip per subsystem — enough to
prove the wire works, cheap enough to run every time.

**Guard recalibration is manual after corpus changes.** The topicality
threshold is a distance in embedding space; re-embedding or adding a doc set
shifts the whole distribution. `calibrate_guard.py` measures the new
distribution and *suggests* a cutoff, but a human commits the constant — where
to draw the on-topic/off-topic line is a policy call, and the script prints the
overlap so you can see when the groups aren't cleanly separable and refuse to
split the difference blindly.

## Tool & library choices

- **`argparse` + `main() -> int` + `sys.exit(main())`** everywhere. Exit codes
  are load-bearing: they make each script a CI gate. `smoke*.py`,
  `coverage_diff.py`, and the batch jobs return non-zero on failure so a
  workflow step goes red. The batch jobs treat partial success as success
  (resumable) and total failure as loud (`return 0 if written else 1`).
- **`python-dotenv`** — every script that touches a credential calls
  `load_dotenv()` first, so a local `.env` and CI secrets are configured the
  same way.
- **Reuse of the real modules, not reimplementation.** The scripts import the
  exact code the app uses: `agent.guard`/`agent.loop` (ask), `index.retrieve`
  (search + calibrate), `index.embed`/`ingest.*` (build), `agent.llm._raw_completion`
  for the batch LLM calls. The batch jobs therefore ride the same provider
  dispatch and fallback chain (OpenRouter/hy3 → Gemini) the workflows configure
  via env — nothing is stubbed, so a CLI green means the production path is green.
- **`gradio_client`** in `smoke_space.py` to hit the Space through its real
  public API, tolerating the `hf_token`→`token` kwarg rename across versions.

## File by file

- **`__init__.py`** — empty; makes `scripts` an importable package so the two
  batch jobs can share code.
- **`ask.py`** — one question end-to-end from the CLI. Guards the input first
  (bails with the guard's reason if it fails), then runs `answer_agentic` and
  pretty-prints the answer, citations (title › anchor + URL), and referrals.
- **`build_index.py`** — the overnight-safe full pipeline. Fails fast if
  `NEON_URL` is missing (unless `--skip-embed`, which never touches Neon).
  Stamps an `index_version` from the crawl timestamp. `--skip-crawl` re-embeds
  the existing snapshot; `--skip-embed` refreshes the snapshot without embedding.
- **`calibrate_guard.py`** — reads the 100/100 eval question files plus inline
  borderline probes, measures `top_distance` per question, prints sorted
  distances + per-group stats, and suggests a threshold (or flags an overlap).
  Print-only by design.
- **`coverage_diff.py`** — set-difference of two committed jsonl dumps; reports
  site pages missing from the index, bucketed by library. Returns 1 if either
  dump is absent. No network, no DB.
- **`dump_docs_inventory.py`** — reads each seed's Sphinx `objects.inv` (kept
  roles: classes/functions/methods/attributes/data + `std:doc`) and sitemap,
  de-dups, and writes the external ground-truth inventory. Must run where
  docs.pytorch.org is reachable.
- **`dump_index_manifest.py`** — one SQL pass over the `chunks` table, rolled up
  per page (title/kind/library + up to 40 headings + chunk count), written as
  the internal manifest. Needs `NEON_URL`; run from Actions.
- **`generate_glosses.py`** — the batch-job home base: `api_pages` (snapshot →
  api pages, core first), batched LLM calls, `parse_glosses` (tolerant JSON
  extraction), `existing_urls_of`/resume, and `git_checkpoint`/`--push`. Writes
  `index/glosses.jsonl`.
- **`generate_questions.py`** — the QuOTE-style twin. Imports the scaffolding
  from `generate_glosses.py` and only differs in prompt, parser
  (`parse_questions`), batch size, and output (`index/questions.jsonl`).
- **`search.py`** — retrieval acceptance test. `retrieve(..., debug=True)`, print
  ranked pointers, hydrate and snippet the top hit. Returns 1 if the index is
  empty. No LLM.
- **`smoke.py`** — four preflight checks (Neon, Gemini, local embedding, optional
  Anthropic), each catching its own exception so one broken connection reports
  instead of crashing the run. Exits non-zero if any required check fails.
- **`smoke_space.py`** — post-deploy health check: poll the HF runtime API to
  RUNNING (or a failure stage), call the Gradio `/respond` endpoint, and
  fail on error markers in the answer. Empty-index → warn, not fail.

## Related docs

- [`../design-content-and-agent-flow.md`](../design-content-and-agent-flow.md) — the system these scripts drive (pipeline, agent tools, session flow).
- [`../deploy-hf-spaces.md`](../deploy-hf-spaces.md) — the deploy that `smoke_space.py` verifies.
- [`../index/README.md`](../index/README.md) — `embed`/`retrieve`/`hydrate`, called by `build_index.py`, `search.py`, `calibrate_guard.py`.
- [`../agent/README.md`](../agent/README.md) — `guard`/`loop`/`llm`, called by `ask.py` and the batch jobs.
