"""index/freshness — the stale-while-revalidate pass, with a fake connection.

No network, no Postgres, no embedding model: _live_units and embed_texts are
monkeypatched and a fake conn records the SQL. The contracts worth pinning:
drifted chunks are healed COMPLETELY (content + content_hash + embedding +
tsv together — a partial write would leave a vector describing text that no
longer exists), only drifted chunks are touched, the TTL stops repeat
fetches, and every failure is swallowed — freshness may never take an answer
down.
"""

from types import SimpleNamespace

import pytest

from index import freshness
from index.embed import chunk_key

URL = "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html"
PAGE_HASH = "livehash123"


def _unit(content, heading=("torch.optim.SGD",)):
    return {
        "url": URL,
        "anchor": "sgd",
        "heading_path": list(heading),
        "content": content,
        "content_hash": PAGE_HASH,  # _live_units stamps the live page's hash
    }


class FakeConn:
    def __init__(self, rows):
        self.rows = rows  # [(chunk_key, content)]
        self.updates: list = []
        self.committed = False
        self.sqls: list[str] = []

    def execute(self, sql, params=None):
        self.sqls.append(sql)
        return SimpleNamespace(fetchall=lambda: self.rows)

    def cursor(self):
        outer = self

        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params):
                outer.sqls.append(sql)
                outer.updates.append(params)

        return Cur()

    def commit(self):
        self.committed = True


@pytest.fixture(autouse=True)
def _fresh_ttl_table():
    freshness.reset()
    yield
    freshness.reset()


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    # a freshness heal re-embeds in-process; tests must never load the 130MB model
    monkeypatch.setattr("index.embed.embed_texts", lambda texts: [[0.1, 0.2]] * len(texts))


def test_drifted_chunks_are_healed_completely_and_in_place(monkeypatch):
    same = _unit("unchanged text")
    drifted = _unit("NEW text", heading=("torch.optim.SGD", "params"))
    monkeypatch.setattr(freshness, "_live_units", lambda url: [same, drifted])
    conn = FakeConn([(chunk_key(same), "unchanged text"), (chunk_key(drifted), "OLD text")])

    assert freshness.refresh_pages([URL], conn=conn) == {URL}
    assert conn.committed
    # only the drifted row, healed whole: content + page hash + fresh vector +
    # the indexed_text the tsv is rebuilt from — never a partial write that
    # would leave the embedding describing text that no longer exists
    assert len(conn.updates) == 1
    content, page_hash, vector, text, key = conn.updates[0]
    assert content == "NEW text" and page_hash == PAGE_HASH and key == chunk_key(drifted)
    assert vector == str([0.1, 0.2])
    assert "NEW text" in text  # indexed_text is built from the NEW content
    heal_sql = next(s for s in conn.sqls if s.lstrip().lower().startswith("update"))
    for column in ("content", "content_hash", "embedding", "tsv"):
        assert column in heal_sql


def test_refresh_ttl_skips_a_recently_checked_url(monkeypatch):
    calls = []
    monkeypatch.setattr(freshness, "_live_units", lambda url: calls.append(url) or [])
    conn = FakeConn([])

    freshness.refresh_pages([URL], conn=conn)
    freshness.refresh_pages([URL], conn=conn)  # within the TTL → no second fetch
    assert calls == [URL]


def test_refresh_new_live_sections_are_left_to_the_crawl(monkeypatch):
    # a section that exists live but not in the DB is a STRUCTURAL change —
    # new/deleted/reorganized pages (and their gloss/question enrichment) are
    # the periodic Build Index's job, not the per-question heal's
    monkeypatch.setattr(freshness, "_live_units", lambda url: [_unit("brand new section")])
    conn = FakeConn([("some-other-key", "old")])

    assert freshness.refresh_pages([URL], conn=conn) == set()
    assert conn.updates == []


def test_refresh_fetch_failure_is_swallowed(monkeypatch):
    def boom(url):
        raise RuntimeError("docs.pytorch.org down")

    monkeypatch.setattr(freshness, "_live_units", boom)
    assert freshness.refresh_pages([URL], conn=FakeConn([])) == set()


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("TORCHDOCS_FRESHNESS", "0")
    assert not freshness.enabled()
    monkeypatch.delenv("TORCHDOCS_FRESHNESS")
    assert freshness.enabled()
