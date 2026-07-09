"""Question generation: the pure parts — prompt shape, reply parsing, resume."""

import json

from scripts.generate_glosses import existing_urls_of
from scripts.generate_questions import (
    QUESTION_MAX_CHARS,
    QUESTIONS_PER_PAGE,
    batch_prompt,
    parse_questions,
)


def _page(url, title="T", excerpt="Applies a linear transformation."):
    return {"url": url, "title": title, "excerpt": excerpt}


def test_batch_prompt_numbers_pages_and_uses_symbol():
    prompt = batch_prompt(
        [
            _page("https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html"),
            _page("https://docs.pytorch.org/docs/stable/amp.html", title="AMP page"),
        ]
    )
    assert "### 0" in prompt and "### 1" in prompt
    assert "torch.nn.Linear" in prompt  # symbol derived from the url
    assert "AMP page" in prompt  # no symbol in the url → the title stands in


def test_parse_questions_plain_json():
    raw = json.dumps(
        [
            {"i": 0, "questions": ["How do I add a fully-connected layer?", "  spaced  "]},
            {"i": 1, "questions": ["What splits a dataset randomly?"]},
        ]
    )
    out = parse_questions(raw, 2)
    assert out[0][0] == "How do I add a fully-connected layer?"
    assert out[0][1] == "spaced"
    assert out[1] == ["What splits a dataset randomly?"]


def test_parse_questions_survives_fences_and_prose():
    raw = 'Sure!\n```json\n[{"i": 0, "questions": ["q1"]}]\n```\nDone.'
    assert parse_questions(raw, 1) == {0: ["q1"]}


def test_parse_questions_drops_malformed_items():
    raw = json.dumps(
        [
            {"i": 5, "questions": ["out of range"]},  # index beyond the batch
            {"i": 0, "questions": []},  # empty list → dropped
            {"i": 1, "questions": ["ok", 7, ""]},  # non-strings/blanks filtered
            "not a dict",
        ]
    )
    assert parse_questions(raw, 2) == {1: ["ok"]}


def test_parse_questions_caps_count_and_length():
    many = [f"question {i}?" for i in range(QUESTIONS_PER_PAGE + 4)]
    long = "x" * (QUESTION_MAX_CHARS + 50)
    out = parse_questions(json.dumps([{"i": 0, "questions": many + [long]}]), 1)
    assert len(out[0]) == QUESTIONS_PER_PAGE
    assert all(len(q) <= QUESTION_MAX_CHARS for q in out[0])


def test_parse_questions_non_json_is_empty():
    assert parse_questions("the pages look great", 1) == {}


def test_existing_urls_of_resumes_from_the_output_file(tmp_path):
    path = tmp_path / "questions.jsonl"
    assert existing_urls_of(path) == set()  # no file yet → nothing covered
    path.write_text(
        '{"url": "https://a", "questions": ["q"]}\n\n{"url": "https://b", "questions": ["q"]}\n'
    )
    assert existing_urls_of(path) == {"https://a", "https://b"}
