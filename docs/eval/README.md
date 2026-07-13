---
title: "eval — the evaluation harness"
kind: reference
package: eval
---

# `eval/` — the measurement layer

The `eval/` package is how we know whether a change to the agent made things better or worse. It holds the static checks that run on every answer, the offline benchmark suites (retrieval / judge / agentic / v0), a retrieval microscope for debugging misses, and the labeled question sets and result files that make "before vs after" a number instead of an opinion.

## Why this package exists / its boundary

The plan committed to **eval from day one** (`PLAN.md` §1.3, before any grounding was built). That order is deliberate: this project's whole reason to exist is that an ungrounded LLM confidently invents PyTorch APIs, and the only honest way to justify the grounding work — retrieval, glosses, reranking, citation-scoping, static-check regeneration — is to *measure* the hallucination it removes. The `eval/hallucinations.md` log is the origin point: 15 v0 questions run with no retrieval, 5 documented invented-API / wrong-signature findings, each one a measurable target that grounding then has to erase.

So the boundary of this package is **measurement, not machinery**. Nothing here answers a user's question in production; it exercises the code that does (`agent/`, `index/`) and scores the output. The one deliberate exception is `checks.py`, which is imported by the live answer path — see [The static checks](#the-static-checks). Everything else reads from the same database and LLM providers the app uses, runs offline (locally or in the `Eval` GitHub Action), and writes JSONL to `eval/results/` for diffing.

## The suites

Each runner is a standalone `python -m eval.<runner>` entry point. They measure different layers so a regression can be localized — retrieval, generated prose, and the full agent loop are scored separately rather than as one opaque end-to-end number.

| Runner | Layer measured | Metric emitted | Results file | In `Eval` workflow |
|---|---|---|---|---|
| `run_v0.py` | Generation + static checks on the manual v0 set | pass/fail table across parses/imports/symbols; `--grounded` adds `grounded_api_rate` + avg citations | `results/v0.jsonl`, `results/v0-grounded.jsonl` | no (local baseline) |
| `run_retrieval.py` | Retrieval only — no LLM | mean `recall@k`, mean `MRR` | `results/retrieval_<set>.jsonl` | `suite=retrieval`, `retrieval+judge`, `all` |
| `run_judge.py` | Generated answer prose (grounded single-shot) | LLM-as-judge faithfulness / answer-relevance / citation-correctness (1–5 → [0,1]) + overall; latency p50/p95/max/mean | `results/judge_<set>.jsonl` | `suite=judge`, `retrieval+judge`, `all` |
| `run_agentic.py` | The full agent loop vs one-shot | citation `coverage` and the **agentic − single-shot delta** | `results/agentic_v1.jsonl` | `suite=agentic`, `all` |
| `diagnose_retrieval.py` | (debugging, not a benchmark) | per-miss rank/distance triage, printed | none (stdout) | runs after `suite=retrieval` |

Notes on the workflow (`.github/workflows/eval.yml`): the default one-click suite is `retrieval+judge` — recall/MRR plus the answer-quality anchor — chosen deliberately because the GitHub mobile app can't set workflow inputs, so the default has to be the measurement worth having. `agentic` stays opt-in (it is slow: many LLM calls per question for both paths, hence the 120-minute job timeout). Every job commits its `results/*.jsonl` back to `main` with `[skip ci]` so before/after diffs live in git history, not in ephemeral CI logs.

A recurring guard shared by the LLM suites: a partial run (`TORCHDOCS_JUDGE_LIMIT`, `TORCHDOCS_AGENTIC_LIMIT`, non-8 `k`) suffixes its results filename (`_first10`, `_k4`) so a bounded CI run can **never masquerade as the full-set result** it would otherwise overwrite.

## The static checks

`checks.py` runs three checks on every `Answer`, and it runs no code — ever. This is a docs assistant, not a sandbox; the checks are static-analysis only:

| Check | What it enforces |
|---|---|
| `parses` | Every ` ```python ` block in `answer_md` passes `ast.parse` (untagged fences, and `pycon` console sessions with `>>>` prompts, are intentionally skipped) |
| `imports` | Every import in those blocks resolves to a torch-family root (`torch`/`torchvision`/`torchaudio`) or the stdlib (`sys.stdlib_module_names`); relative imports are always rejected |
| `symbols` | Every entry in the answer's `symbols_used` actually appears in the prose, tolerant of conventional spelling (`nn.Linear` for `torch.nn.Linear`, `F.relu`, `.add_`) with word-boundary matching so `torch.relu` isn't "found" inside `prelu` |

**The important fact: these checks run in the *live* answer path, not just offline.** `agent/grounded.py` imports `run_checks` and calls it inside `_regenerate_if_checks_fail`: when a generated answer fails a check (unparseable snippet, off-family import, a symbol listed but missing from the prose), it re-asks the model *once* with the specific failure reasons injected, and keeps the repair only if it is strictly cleaner — never blocking the user on a failed check. So the same code that scores answers in the benchmark is a real-time quality gate on production answers. That dual use is why `checks.py` is pure functions over an `Answer` with no I/O: it has to be cheap and side-effect-free enough to sit in the request path.

## LLM-as-judge

`run_judge.py` closes the gap the other suites leave open: retrieval eval scores whether the right *pages* were found and the static checks score whether *code* parses, but neither scores the prose the user actually reads. The judge does, on three dimensions:

- **faithfulness** — is every claim supported by the provided context, or invented beyond it? (an honest "not in the docs" referral counts as faithful, not a failure). This is the hallucination axis the grounding contract exists to hold.
- **answer_relevance** — does the answer address the question actually asked?
- **citation_correctness** — do the cited sections genuinely support the claims, and is every load-bearing claim cited?

Each is scored 1–5 and normalized to `[0,1]` so it shares a scale with the retrieval metrics; the mean is the `overall` before/after number. The judge sees the **same numbered context the answer saw** (`build_context`), so faithfulness is checked against the real inputs rather than a re-retrieval.

Two caveats are baked into the docstring, honestly:

1. **Same-family bias.** On free-tier keys the judge may be the *same model* that wrote the answer, which biases toward leniency — so the score is a **relative gauge for regressions, not an absolute grade**. Pointing `TORCHDOCS_*` at a stronger, independent judge model is an open `PLAN.md` M4 item ("Pick a dedicated judge model"); until it lands, read the numbers as a trend, not a grade.
2. **Trust boundary.** The judge reads model-written answers and doc text, so its system prompt is hardened to *score* those as data, never to *follow* embedded instructions — but a suspiciously perfect run still deserves skepticism.

The judge run also captures **latency**, and deliberately times only what the user waits on — retrieval + answer generation, with the eval-only judge call excluded — then reports p50/p95/max/mean. That is the core UX number: question in → answer out.

## Design decisions & rationale

- **Static before semantic.** The checks in `checks.py` (deterministic, free, in-path) run before any LLM judgment. A parse error or a fabricated import is a hard defect that never needs a model to adjudicate, and catching it deterministically keeps the expensive, noisier judge focused on prose quality. The regeneration loop reflects the same ordering: cheap static repair first, LLM judgment reserved for offline scoring.
- **Before/after comparability via `index_version`.** The corpus is always-latest (a PyTorch release just shows up as a hash-diff on the next recrawl), so "recall went up" is only meaningful against a fixed index. `index_version` is an internal crawl-build id kept precisely for eval comparability (and cache invalidation), decoupled from PyTorch version numbers — it's what lets a retrieval delta be attributed to a retrieval change rather than a corpus change underneath it.
- **A labeled retrieval set.** `run_retrieval.py` measures recall/MRR against questions carrying *expected* source groups (each group a list of alternative URL/title substrings, any alternative counting as a hit). This costs authoring effort — 100 questions written against the verified docs inventory — but it's what makes retrieval measurable without an LLM in the loop, which is what makes it fast, deterministic, and cheap enough to run on every retrieval-affecting change. The glosses×reranker recall jump from 0.430 → 0.840 (documented in `docs/retrieval-gaps-and-improvements.md`) is only a claim because this labeled set exists.
- **Coverage delta, not absolute, for the agent loop.** `run_agentic.py`'s headline is `agentic_coverage − single_shot_coverage`: catalog/compare/recipe answers are spread across pages, so the honest question isn't "is the agentic answer good" but "did the loop assemble *more* of the answer than one search would." A negative delta is reported as an honest negative result, not hidden.

## Tool & library choices

- **`ast` for the static checks.** Parsing candidate code with the standard-library `ast` module (not regex, not execution) is both safe — no code runs — and precise: `ast.walk` finds every `Import`/`ImportFrom` node including relative imports, which a regex would miss. `textwrap.dedent` first, because models routinely indent whole blocks inside markdown lists.
- **The app's own LLM dispatch, reused for judging.** `run_judge.py` calls `agent.llm._raw_completion` with the shared provider/fallback chain rather than a bespoke client. Same dispatch, same fallback behavior, one place to configure providers — and the judge reply is validated through Pydantic models (`JudgeScores`), with `_extract_json` tolerating the fences and stray prose models emit despite an "only JSON" instruction.
- **JSONL result files.** Every runner writes one JSON object per line to `eval/results/`. It's append-friendly, diffs cleanly in git (which is where before/after lives), and is trivial to load back for aggregation. Runners that spend scarce free-tier quota (`run_v0.py`, and the LLM suites) flush after every question and, in `run_v0.py`, resume from prior results unless `--fresh` — so a crash or rate-limit mid-run never discards answers already paid for.

## File by file

- **`__init__.py`** — empty; marks `eval/` as a package so the runners are importable as `eval.<name>`.
- **`checks.py`** — the three static checks (`check_code_parses`, `check_imports_allowed`, `check_symbols_present`), the `CHECKS` registry, `run_checks`, and `format_table` for the pass/fail grid. Pure, no I/O; imported by both the runners and the live `agent/grounded.py` regeneration path.
- **`run_v0.py`** — runs the 15-question manual v0 set through `answer_question` (ungrounded) or, with `--grounded`, through `answer_grounded`, applies `run_checks`, and prints the pass table. `--grounded` additionally computes `grounded_api_rate` (share of `symbols_used` that actually exist in the docs index, via a tsvector probe) and average citations. Resumable; flushes per question.
- **`run_retrieval.py`** — the retrieval-only benchmark. Loads the selected set (`v1` inline expectations, or `v0` with a `retrieval_v0.jsonl` sidecar), runs `index.retrieve.retrieve` at `k`, and computes per-question `recall@k` and `MRR`, then aggregates. No LLM. The reference before/after suite for any retrieval change.
- **`run_judge.py`** — the LLM-as-judge suite (above): generates a grounded single-shot answer, scores it on the three dimensions with a hardened judge prompt, normalizes to `[0,1]`, aggregates, and reports UX latency percentiles. Pure parts (`_normalize`, `_extract_json`, `parse_judge_reply`, `aggregate`) are unit-tested.
- **`run_agentic.py`** — the agent-loop benchmark for multi-page (catalog/compare/recipe) questions in `agentic_v1.jsonl`. Runs each through both `answer_agentic` and `answer_grounded`, scores citation `coverage` against `expected_any` (objective URL/anchor/title substring match — no judge), and reports the loop-vs-single-shot delta.
- **`diagnose_retrieval.py`** — a debugging microscope, not a benchmark. For a handful of known descriptive misses it prints, per kind-pool, the nearest candidates with cosine distances and locates the *expected* page in the raw dense/keyword candidates — triaging whether a miss was never a candidate, out-ranked within its pool, or dropped by the relevance-gap filter, and whether the expected page's own nearest chunk is a *crowding* problem (deeper pool helps) or an *embedding* problem (only doc-side enrichment helps). Runs in Actions after the retrieval suite; intended to be deleted once the fix it points at lands.

### Data files under `eval/`

- **`questions_v0.jsonl`** (15) — the manual day-one set spanning the five question types (usage / catalog / recipe / source / edge); driven by `run_v0.py`.
- **`questions_v1.jsonl`** (100) — the main labeled set, authored against the verified external docs inventory with expected sources inline; driven by `run_retrieval.py` and `run_judge.py`.
- **`retrieval_v0.jsonl`** — expected-source sidecar for the v0 set (v1 keeps expectations inline).
- **`agentic_v1.jsonl`** (20) — the multi-page catalog/compare/recipe set with `expected_any` source groups; driven by `run_agentic.py`.
- **`invalid_v1.jsonl`** (100) — out-of-scope questions (e.g. React/`useState`) for refusal / negative-case calibration, consumed by `scripts/` (e.g. `calibrate_guard.py`), not the core runners.
- **`docs_inventory.jsonl`** / **`index_manifest.jsonl`** — the verified external docs inventory (symbol → URL) and the live index manifest; produced by `scripts/dump_*` and used to author expectations and diff coverage (`scripts/coverage_diff.py`).
- **`hallucinations.md`** — the v0 ungrounded-baseline hallucination log (OKF-style, per-finding frontmatter): the measurable target grounding had to erase.
- **`results/`** — committed JSONL outputs of the suites (`retrieval_*`, `judge_*`, `agentic_*`, `v0*`), the substrate for before/after diffs.

## Related docs

- [`../design-content-and-agent-flow.md`](../design-content-and-agent-flow.md) — the product boundary (site + `main` only, no sandbox, tool-calling loop) the suites are written to measure.
- [`../retrieval-gaps-and-improvements.md`](../retrieval-gaps-and-improvements.md) — the RAG maturity review whose recall/MRR numbers come straight out of `run_retrieval.py` and `diagnose_retrieval.py`.
- [`../agent/README.md`](../agent/README.md) — the sibling package these suites exercise (`answer_grounded`, `answer_agentic`, and the `checks.py` regeneration hook).
