"""index/freshness — the stale-while-revalidate pass, with a fake connection.

No network, no Postgres: _live_units is monkeypatched and a fake conn records
the SQL. The contracts worth pinning: only drifted chunks are updated, the
UPDATE touches content ONLY (content_hash/embedding must stay so the weekly
crawl re-embeds properly), the TTL stops repeat fetches, and every failure is
swallowed — freshness may never take an answer down.
"""

from types import SimpleNamespace

import pytest

from index import freshness
from index.embed import chunk_key

URL = "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html"


def _unit(content, heading=("torch.optim.SGD",)):
    return {"url": URL, "anchor": "sgd", "heading_path": list(heading), "content": content}


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

            def executemany(self, sql, params):
                outer.sqls.append(sql)
                outer.updates.extend(params)

        return Cur()

    def commit(self):
        self.committed = True


@pytest.fixture(autouse=True)
def _fresh_ttl_table():
    freshness.reset()
    yield
    freshness.reset()


def test_refresh_updates_only_drifted_chunks_and_never_the_hash(monkeypatch):
    same = _unit("unchanged text")
    drifted = _unit("NEW text", heading=("torch.optim.SGD", "params"))
    monkeypatch.setattr(freshness, "_live_units", lambda url: [same, drifted])
    conn = FakeConn([(chunk_key(same), "unchanged text"), (chunk_key(drifted), "OLD text")])

    assert freshness.refresh_pages([URL], conn=conn) == {URL}
    assert conn.updates == [("NEW text", chunk_key(drifted))]  # only the drifted row
    assert conn.committed
    # the embedding's provenance must survive: content is the ONLY column written
    update_sql = next(s for s in conn.sqls if s.lstrip().lower().startswith("update"))
    assert "content_hash" not in update_sql and "embedding" not in update_sql


def test_refresh_ttl_skips_a_recently_checked_url(monkeypatch):
    calls = []
    monkeypatch.setattr(freshness, "_live_units", lambda url: calls.append(url) or [])
    conn = FakeConn([])

    freshness.refresh_pages([URL], conn=conn)
    freshness.refresh_pages([URL], conn=conn)  # within the TTL → no second fetch
    assert calls == [URL]


def test_refresh_new_live_sections_are_left_to_the_crawl(monkeypatch):
    # a section that exists live but not in the DB has no embedding to serve
    # under — an in-place update cannot represent it, so it must be skipped
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
