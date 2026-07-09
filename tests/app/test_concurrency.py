"""The web app answers many questions at once, and each answer is isolated.

Two things make the app concurrent: serve() opens the Gradio queue past its
serial default, and respond() carries no shared state so parallel calls can't
bleed into each other. Both are pinned here without launching a real server.
"""

import concurrent.futures as cf
import importlib
import time

import pytest

from agent.schemas import Answer
from app import main


@pytest.fixture(autouse=True)
def _disable_guard(monkeypatch):
    # the guard would try to load a real classifier here (it is exercised in
    # tests/agent/test_guard.py); keep these tests about concurrency only
    monkeypatch.setenv("TORCHDOCS_GUARD", "0")


class FakeDemo:
    """Records the kwargs serve() passes to queue()/launch() — no server."""

    def __init__(self):
        self.calls = {}

    def queue(self, **kwargs):
        self.calls["queue"] = kwargs
        return self

    def launch(self, **kwargs):
        self.calls["launch"] = kwargs


def test_serve_opens_queue_and_lifts_thread_pool():
    demo = FakeDemo()
    main.serve(demo)

    # the queue is opened to CONCURRENCY workers, not Gradio's serial default of 1
    assert demo.calls["queue"]["default_concurrency_limit"] == main.CONCURRENCY
    # the worker pool is lifted in step so it never caps below CONCURRENCY
    assert demo.calls["launch"]["max_threads"] >= main.CONCURRENCY
    assert demo.calls["launch"]["server_name"] == "0.0.0.0"


def test_concurrency_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("TORCHDOCS_CONCURRENCY", "3")
    try:
        importlib.reload(main)
        assert main.CONCURRENCY == 3
    finally:
        monkeypatch.delenv("TORCHDOCS_CONCURRENCY", raising=False)
        importlib.reload(main)  # restore the default for the rest of the suite


def test_respond_runs_in_parallel_without_state_bleed(monkeypatch):
    # a stand-in agent that sleeps (like a real LLM/DB wait) and echoes its input
    def fake_agent(question, **kwargs):
        time.sleep(0.2)
        return Answer(answer_md=f"echo:{question}", torch_version="unknown")

    monkeypatch.setattr(main, "answer_routed", fake_agent)

    questions = [f"question number {i}" for i in range(8)]
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(main.respond, questions))
    elapsed = time.perf_counter() - start

    # isolation: every caller got back its OWN question, none crossed wires
    for question, output in zip(questions, outputs, strict=True):
        assert f"echo:{question}" in output
    # parallelism: 8 × 0.2s overlapped (~0.2s), nowhere near the 1.6s serial sum
    assert elapsed < 1.0


def test_queue_is_bounded():
    # backpressure: a flood gets "queue full" instead of an ever-growing line
    demo = FakeDemo()
    main.serve(demo)
    assert demo.calls["queue"]["max_size"] == main.QUEUE_SIZE
