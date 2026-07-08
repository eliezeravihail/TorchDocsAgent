"""The answer-quality judge's pure parts — parsing/scaling/aggregation, no LLM."""

from types import SimpleNamespace

import pytest

from eval.run_judge import (
    DIMENSIONS,
    JudgeScores,
    aggregate,
    build_judge_prompt,
    judge_answer,
    normalized_scores,
    parse_judge_reply,
)


def _reply(f, r, c):
    return (
        f'{{"faithfulness": {{"score": {f}, "why": "x"}}, '
        f'"answer_relevance": {{"score": {r}, "why": "y"}}, '
        f'"citation_correctness": {{"score": {c}, "why": "z"}}}}'
    )


def test_parse_plain_json():
    scores = parse_judge_reply(_reply(5, 4, 3))
    assert scores.faithfulness.score == 5
    assert scores.answer_relevance.score == 4
    assert scores.citation_correctness.score == 3


def test_parse_survives_code_fence_and_prose():
    reply = "Sure, here is my verdict:\n```json\n" + _reply(4, 4, 4) + "\n```\nHope that helps!"
    assert parse_judge_reply(reply).faithfulness.score == 4


def test_parse_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        parse_judge_reply(_reply(7, 4, 3))  # 7 is outside 1–5


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_judge_reply("the answer looks great to me")


def test_normalize_maps_1_5_to_unit_interval():
    scores = JudgeScores.model_validate_json(_reply(1, 3, 5))
    norm = normalized_scores(scores)
    assert norm["faithfulness"] == 0.0  # 1 → 0
    assert norm["answer_relevance"] == 0.5  # 3 → 0.5
    assert norm["citation_correctness"] == 1.0  # 5 → 1
    assert norm["overall"] == pytest.approx((0.0 + 0.5 + 1.0) / 3)


def test_prompt_frames_question_context_and_citations():
    answer = SimpleNamespace(
        answer_md="Use torch.optim.SGD.",
        citations=[SimpleNamespace(url="https://d/sgd", anchor="sgd", title="SGD")],
    )
    prompt = build_judge_prompt("How do I use SGD?", "[1] TITLE: SGD\n...", answer)
    assert "How do I use SGD?" in prompt
    assert "[1] TITLE: SGD" in prompt
    assert "https://d/sgd#sgd (SGD)" in prompt
    assert "Use torch.optim.SGD." in prompt


def test_prompt_handles_no_citations_and_no_context():
    answer = SimpleNamespace(answer_md="I could not find this.", citations=[])
    prompt = build_judge_prompt("q", "", answer)
    assert "(none)" in prompt  # citations
    assert "(none retrieved)" in prompt  # context


def test_aggregate_means_only_over_scored_records():
    records = [
        {"id": "a", "scores": {d: 1.0 for d in (*DIMENSIONS, "overall")}},
        {"id": "b", "scores": {d: 0.0 for d in (*DIMENSIONS, "overall")}},
        {"id": "c", "error": "boom"},  # unscored → excluded from the mean
    ]
    agg = aggregate(records)
    assert agg["faithfulness"] == 0.5
    assert agg["overall"] == 0.5


def test_aggregate_empty_when_nothing_scored():
    assert aggregate([{"id": "a", "error": "boom"}]) == {}


def test_judge_answer_uses_the_raw_completion_path():
    """judge_answer wires prompt → _raw_completion → parse; a fake client stands in."""
    captured = {}

    class FakeCompat:
        class chat:  # noqa: N801 — mimic the openai client surface
            class completions:
                @staticmethod
                def create(model, messages, **kwargs):
                    captured["system"] = messages[0]["content"]
                    captured["user"] = messages[1]["content"]
                    body = _reply(5, 5, 4)
                    msg = SimpleNamespace(content=body)
                    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    answer = SimpleNamespace(answer_md="A.", citations=[])
    scores = judge_answer("q?", "[1] ctx", answer, provider="openai-compat", client=FakeCompat())
    assert scores.citation_correctness.score == 4
    assert "q?" in captured["user"] and "[1] ctx" in captured["user"]
