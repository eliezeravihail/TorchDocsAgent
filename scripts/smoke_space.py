"""Post-deploy smoke test for the Hugging Face Space.

Runs in GitHub Actions (which can reach both the Space and OpenRouter) after
the sync-to-hf workflow finishes — a live health check the local dev sandbox
can't perform because its egress is locked down. Steps:

  1. poll the HF runtime API until the Space finishes (re)building and is RUNNING
  2. ask it a real question through the Gradio API (api_name="/respond")
  3. fail the job if the answer is an LLM/transport error, so a broken deploy
     shows up as a red check instead of silently serving errors

Env:
  SPACE_ID        repo id of the Space           (default eliezeravihail/torchdocs-agent)
  HF_TOKEN        token for the runtime API / private space (optional for public)
  SMOKE_QUESTION  question to ask                 (has a sensible default)
  BUILD_TIMEOUT   seconds to wait for RUNNING     (default 900)
"""

from __future__ import annotations

import os
import sys
import time

import requests

SPACE_ID = os.environ.get("SPACE_ID", "eliezeravihail/torchdocs-agent")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
# `or` (not a get default): the workflow always sets SMOKE_QUESTION, but it's
# empty on the workflow_run trigger — an empty string must fall back too
QUESTION = os.environ.get("SMOKE_QUESTION") or "How do I use torch.optim.SGD with momentum?"
BUILD_TIMEOUT = int(os.environ.get("BUILD_TIMEOUT", "900"))
POLL_EVERY = 15
RUNTIME_URL = f"https://huggingface.co/api/spaces/{SPACE_ID}/runtime"

# markers that mean generation/transport actually broke → hard fail
LLM_ERROR_MARKERS = (
    "went wrong",
    "connection error",
    "llm call failed",
    "all providers failed",
    "is not set",
    "must be a full http",
)
# a working app that simply has no index yet → warn, don't fail (different subsystem)
EMPTY_INDEX_MARKER = "could not find anything"


def _headers() -> dict:
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def wait_for_running() -> bool:
    """Block until the Space reports RUNNING, or a failure/timeout."""
    deadline = time.time() + BUILD_TIMEOUT
    # small grace so we poll the *new* build, not the old RUNNING instance that
    # is still up for the moment right after the push
    time.sleep(30)
    last = None
    while time.time() < deadline:
        try:
            resp = requests.get(RUNTIME_URL, headers=_headers(), timeout=30)
            stage = resp.json().get("stage")
        except Exception as exc:  # noqa: BLE001 — transient during rebuild
            stage = f"unreachable ({exc})"
        if stage != last:
            print(f"[smoke] space stage: {stage}", flush=True)
            last = stage
        if stage == "RUNNING":
            return True
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            print(f"[smoke] space entered failure stage: {stage}", flush=True)
            return False
        time.sleep(POLL_EVERY)
    print(f"[smoke] timed out after {BUILD_TIMEOUT}s waiting for RUNNING", flush=True)
    return False


def ask_space() -> str:
    """Call the Space's /respond endpoint and return the answer text."""
    from gradio_client import Client

    # the token kwarg was renamed across gradio_client versions (hf_token → token)
    # and the Space is public anyway, so try the variants and fall back tokenless
    client = None
    for kw in ("token", "hf_token"):
        try:
            client = Client(SPACE_ID, **{kw: HF_TOKEN})
            break
        except TypeError:
            continue
    if client is None:
        client = Client(SPACE_ID)
    result = client.predict(QUESTION, api_name="/respond")
    return result if isinstance(result, str) else str(result)


def main() -> int:
    print(f"[smoke] space={SPACE_ID} question={QUESTION!r}", flush=True)
    if not wait_for_running():
        return 1

    # the server just came up; give _warm_up (embedding model load) a moment
    time.sleep(10)
    try:
        answer = ask_space()
    except Exception as exc:  # noqa: BLE001 — the call itself failing is a fail
        print(f"[smoke] FAIL: calling the Space raised {type(exc).__name__}: {exc}", flush=True)
        return 1

    print("[smoke] ---------------- answer ----------------", flush=True)
    print(answer[:2000], flush=True)
    print("[smoke] ------------------------------------------", flush=True)

    low = answer.lower()
    hit = next((m for m in LLM_ERROR_MARKERS if m in low), None)
    if hit:
        print(f"[smoke] FAIL: answer contains an error marker: {hit!r}", flush=True)
        return 1
    if not answer.strip():
        print("[smoke] FAIL: empty answer", flush=True)
        return 1
    if EMPTY_INDEX_MARKER in low:
        # generation worked; the corpus/index is just empty — surface loudly
        # but don't red the deploy, that's a separate (indexing) concern
        print("[smoke] WARNING: LLM ok but the docs index looks empty", flush=True)
    print("[smoke] PASS: Space answered without an LLM/transport error", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
