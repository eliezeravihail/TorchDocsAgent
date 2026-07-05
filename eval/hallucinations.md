---
kind: hallucination-log
eval_version: v0
date: 2026-07-05
models: [gemini-2.5-flash, gemini-2.5-flash-lite, "poolside/laguna-xs-2.1:free"]
answered: 15/15  # full single-model run via OpenRouter (laguna) on Actions; earlier partial Gemini run kept for findings 1-3
static_checks_pass: 10/15
purpose: >
  Ungrounded-baseline findings (M1: no retrieval). Each finding is a measurable
  target for M2 grounding — rerun the same questions with retrieval and these
  should disappear or carry citations that expose them.
---

# Hallucination log — v0 baseline (no retrieval)

## Finding 1 — Deprecated technology recommended as the current path

```yaml
question_id: q15
torch_version_claimed: "2.2"
severity: high
category: stale-guidance
```

Asked whether one can train directly on an Android phone, the model answered
"Yes, using **PyTorch Mobile**" and recommended converting to TorchScript for
on-device use, adding that PyTorch Mobile "supports fine-tuning and other
training-related operations."

PyTorch Mobile is deprecated; the current on-device story in the docs is
**ExecuTorch**, which the answer never mentions. On-device *training* support
is far more limited than the answer implies. A user following this walks into
a dead-end toolchain. This is precisely the class of error the always-latest
docs corpus fixes: the ExecuTorch pages are in the v1.1 seed list, and the
edge-question flow (grade → partial answer + referral) would have surfaced
them or honestly deferred.

## Finding 2 — Inconsistent and outdated torch_version claims across answers

```yaml
question_id: [q02, q03, q07, q09, q08, q15]
torch_version_claimed: ["2.1", "2.1", "2.1", "2.1", "2.12", "2.2"]
severity: medium
category: staleness
```

The same model, in the same run, stamped its answers as targeting PyTorch
2.1, 2.2, and 2.12 — whatever its training data happened to anchor on per
question. Current stable is 2.12. Ungrounded answers cannot know "today's"
version; after M2 the version comes from the crawled site (one truth), not
from the model's memory, and the stamp becomes trustworthy.

## Finding 3 — Audio-generation answer omits torchaudio entirely

```yaml
question_id: q09
torch_version_claimed: "2.1"
severity: medium
category: missed-canonical-library
```

Asked how to generate music/audio, the model produced a generic
LSTM/WaveNet-era outline (`symbols_used` contains only `torch.nn.*` and
`torch.optim.Adam`) and never mentions **torchaudio** — the library the
PyTorch site dedicates to audio, which is in our v1 core corpus. The answer
is not factually false, but it fails the product's whole point: routing the
user to what the documentation actually offers. With retrieval, the
decomposed queries land in `docs.pytorch.org/audio/**` and the answer builds
on (and cites) those pages.

## Finding 4 — A fabricated library, recommended by name

```yaml
question_id: q15
model: poolside/laguna-xs-2.1:free
severity: high
category: fabrication
```

Asked about training on Android, the model recommends "**Leverage
LIBTHONG**: A lightweight library for training on Android, compatible with
PyTorch", including cross-compilation instructions — **no such library
exists**. It also invents PyTorch build flags (`USE_MOBILE=1`,
`BUILD_TRAINING=1`). This is the textbook invented-name hallucination the
Gemini runs didn't produce; smaller free models do fabricate. Grounded
answers cannot cite a nonexistent library — the check "every symbol exists
in the index" and the referral discipline both kill this class.

## Finding 5 — Plausible but nonexistent source-file paths

```yaml
question_id: q12
model: poolside/laguna-xs-2.1:free
severity: high
category: fabrication
```

Asked how conv2d is implemented, the model cites specific files:
`aten/src/ATen/native/conv.cpp`, `aten/src/ATen/native/cuda/conv_cublas.cpp`,
`aten/src/ATen/native/cuda/conv_direct.cpp` — none of which exist in the
PyTorch repo (the real entry point is e.g. `Convolution.cpp`). Exactly the
failure mode the `ask_source` tool (DeepWiki) + mandatory referral link is
designed for: source claims must come from, and link to, the real tree.

## Observations for M2

- No fabricated API names surfaced in the 8 answers reviewed — modern models
  hallucinate less by *inventing* symbols and more by *dating*: deprecated
  paths presented as current (Findings 1–2). The grounding metric should
  therefore track staleness, not just `grounded_api_rate`.
- `symbols_used` bookkeeping was honest throughout (after alias-aware
  matching); the static checks pass 8/8. The value of M2 is currency and
  coverage, not syntax.
