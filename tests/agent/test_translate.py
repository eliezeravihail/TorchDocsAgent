from types import SimpleNamespace

from agent.translate import looks_english, translate_to_english


def test_looks_english_ascii_true():
    assert looks_english("how do I use SGD scheduler")
    assert looks_english("torch.optim.lr_scheduler.LinearLR")


def test_looks_english_hebrew_false():
    assert not looks_english("כיצד לבצע סקדולר לינארי לרשת שלי")


def test_english_query_passes_through_without_llm():
    # no client, no keys — must not attempt any call for English input
    assert translate_to_english("linear learning rate scheduler") == (
        "linear learning rate scheduler"
    )


def _fake_openai_client(reply_text):
    message = SimpleNamespace(content=reply_text)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    completions = SimpleNamespace(create=lambda **kw: response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_hebrew_query_is_translated(monkeypatch):
    monkeypatch.setenv("TORCHDOCS_PROVIDER", "openai-compat")
    client = _fake_openai_client("linear learning rate scheduler\n")
    out = translate_to_english(
        "כיצד לבצע סקדולר לינארי לרשת שלי", provider="openai-compat", client=client
    )
    assert out == "linear learning rate scheduler"


def test_translation_failure_falls_back_to_original():
    def boom(**kw):
        raise RuntimeError("upstream 429")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    original = "כיצד לבצע סקדולר"
    # any failure (rate limit, network) must degrade to the original query,
    # never crash the search
    assert translate_to_english(original, provider="openai-compat", client=client) == original
