"""Gloss generation: the pure parts — prompt shape, reply parsing, resume."""

import json

from scripts.generate_glosses import GLOSS_MAX_CHARS, batch_prompt, parse_glosses


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
    assert "AMP page" in prompt  # no dotted symbol → falls back to the title
    assert "2 glosses" in prompt


def test_parse_glosses_happy_path():
    raw = 'Here you go:\n[{"i": 0, "gloss": "The fully-connected layer."}, {"i": 1, "gloss": "x"}]'
    assert parse_glosses(raw, 2) == {0: "The fully-connected layer.", 1: "x"}


def test_parse_glosses_drops_malformed_and_out_of_range_items():
    raw = json.dumps(
        [
            {"i": 0, "gloss": "ok"},
            {"i": 9, "gloss": "index beyond the batch"},
            {"i": 1, "gloss": ""},  # empty
            {"i": "1", "gloss": "non-int index"},
            "not a dict",
        ]
    )
    assert parse_glosses(raw, 2) == {0: "ok"}


def test_parse_glosses_truncates_runaway_gloss():
    raw = json.dumps([{"i": 0, "gloss": "x" * 1000}])
    assert len(parse_glosses(raw, 1)[0]) == GLOSS_MAX_CHARS


def test_parse_glosses_garbage_is_empty_not_crash():
    assert parse_glosses("no json here", 3) == {}
    assert parse_glosses("[not valid json]", 3) == {}
