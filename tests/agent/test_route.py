"""The question router — shape heuristic, path dispatch, and escalation."""

from agent.route import answer_routed, needs_loop
from agent.schemas import Answer, Citation

# --- the heuristic -----------------------------------------------------------

LOOP_QUESTIONS = [
    "What loss functions exist for classification?",
    "Which LR schedulers are available in PyTorch?",
    "List all the pooling layers",
    "What's the difference between CrossEntropyLoss and NLLLoss?",
    "BatchNorm vs LayerNorm — which should I use?",
    "Should I use Adam or SGD for a transformer?",
    "How do I build a CNN to classify images?",
    "Train a sequence model end to end on my own data",
    "How is conv2d implemented?",
    "What happens under the hood when I call backward()?",
]

GROUNDED_QUESTIONS = [
    "How do I use torch.optim.SGD with momentum?",
    "How do I save and load a model checkpoint?",
    "What does torch.gather do?",
    "How do I move a tensor to the GPU?",
    "Why is my loss NaN after a few steps?",
]


def test_multi_source_shapes_go_to_the_loop():
    for q in LOOP_QUESTIONS:
        assert needs_loop(q), f"expected loop for: {q!r}"


def test_usage_questions_stay_single_shot():
    for q in GROUNDED_QUESTIONS:
        assert not needs_loop(q), f"expected grounded for: {q!r}"


# --- dispatch and escalation -------------------------------------------------


def _grounded_answer(cited=True):
    cites = [Citation(url="https://docs.pytorch.org/x")] if cited else []
    return Answer(answer_md="grounded", citations=cites)


def test_simple_question_takes_the_grounded_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.grounded.answer_grounded",
        lambda q, **kw: calls.append("grounded") or _grounded_answer(),
    )
    monkeypatch.setattr(
        "agent.loop.answer_agentic",
        lambda q, **kw: calls.append("loop") or Answer(answer_md="loop"),
    )
    out = answer_routed("How do I use torch.optim.SGD with momentum?")
    assert out.answer_md == "grounded"
    assert calls == ["grounded"]  # the loop was never paid for


def test_catalog_question_takes_the_loop(monkeypatch):
    monkeypatch.setattr(
        "agent.loop.answer_agentic", lambda q, **kw: Answer(answer_md="loop")
    )

    def no_grounded(q, **kw):  # pragma: no cover — must not be called
        raise AssertionError("grounded path must not run for a catalog question")

    monkeypatch.setattr("agent.grounded.answer_grounded", no_grounded)
    assert answer_routed("What loss functions exist for classification?").answer_md == "loop"


def test_uncited_grounded_answer_escalates_to_the_loop(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.grounded.answer_grounded",
        lambda q, **kw: calls.append("grounded") or _grounded_answer(cited=False),
    )
    monkeypatch.setattr(
        "agent.loop.answer_agentic",
        lambda q, **kw: calls.append("loop") or Answer(answer_md="loop"),
    )
    out = answer_routed("How do I use the frobnicate transform?")
    assert out.answer_md == "loop"  # an unsourced reply must not ship
    assert calls == ["grounded", "loop"]


def test_grounded_path_gets_the_question_verbatim(monkeypatch):
    # there is no translation step anymore — the guard bounces non-English input
    # up front, so the router hands the grounded path the question as received
    seen = {}

    def fake_grounded(q, **kw):
        seen["question"] = q
        return _grounded_answer()

    monkeypatch.setattr("agent.grounded.answer_grounded", fake_grounded)
    answer_routed("how do I use SGD with momentum")
    assert seen["question"] == "how do I use SGD with momentum"
