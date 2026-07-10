"""build_index — the resumable embed/upsert/purge pass, with a fake connection.

No live Postgres or embedding model: connect()/ensure_schema/get_meta/set_meta
and iter_corpus_units are faked, and embed_fn is injected. The behaviour worth
pinning is the stale-chunk purge — especially that a recipe bump (which forces
a full re-embed) still purges deleted chunks, the bug this test guards.
"""

from types import SimpleNamespace

import pytest

import index.db as db
import index.embed as embed
from index.embed import EMBED_RECIPE, build_index, chunk_key


def _unit(url, chash="h1"):
    return {
        "url": url, "anchor": "", "heading_path": ["Section"], "page_title": "T",
        "library": "core", "kind": "api", "source_link": "", "content_hash": chash,
        "content": "body text",
    }


class _Result:
    def __init__(self, conn, sql, params):
        self.conn, self.sql = conn, sql
        if sql.strip().lower().startswith("delete"):
            conn.deleted.extend(params[0])

    def fetchall(self):
        if "where content" in self.sql:  # the content-backfill probe — nothing to fill
            return []
        return self.conn.db_rows if "select chunk_key" in self.sql else []

    def fetchone(self):
        return (self.conn.count,) if "count(*)" in self.sql else None


class _CursorCtx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.upserts.append(params)

    def executemany(self, sql, params):
        self.conn.backfilled.extend(params)


class FakeConn:
    def __init__(self, db_rows):
        self.db_rows = list(db_rows)
        self.count = len(db_rows)
        self.deleted: list[str] = []
        self.upserts: list = []
        self.backfilled: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        return _Result(self, sql, params)

    def cursor(self):
        return _CursorCtx(self)

    def commit(self):
        pass


def _patch(monkeypatch, conn, units, recipe):
    monkeypatch.setattr(db, "connect", lambda: conn)
    monkeypatch.setattr(db, "ensure_schema", lambda c: None)
    monkeypatch.setattr(db, "get_meta", lambda c, k: recipe)
    monkeypatch.setattr(db, "set_meta", lambda c, k, v: None)
    monkeypatch.setattr(embed, "iter_corpus_units", lambda corpus_dir=None: iter(units))


def test_recipe_change_still_purges_stale_chunks(monkeypatch):
    live = _unit("https://docs.pytorch.org/docs/stable/a.html")
    conn = FakeConn([(chunk_key(live), "h1"), ("STALE_KEY", "hX")])
    _patch(monkeypatch, conn, [live], recipe="old-recipe")  # != EMBED_RECIPE → full re-embed

    result = build_index("v1", embed_fn=lambda texts: [[0.0, 0.0]] * len(texts))

    assert "STALE_KEY" in conn.deleted  # regression: purge must run even on a recipe bump
    assert conn.upserts  # recipe change re-embeds the live chunk
    assert result["snapshot_chunks"] == 1


def test_unchanged_chunk_not_reembedded_and_stale_purged(monkeypatch):
    live = _unit("https://docs.pytorch.org/docs/stable/a.html")
    conn = FakeConn([(chunk_key(live), "h1"), ("STALE_KEY", "hX")])
    _patch(monkeypatch, conn, [live], recipe=EMBED_RECIPE)  # same recipe → incremental

    build_index("v1", embed_fn=lambda texts: [[0.0, 0.0]] * len(texts))

    assert conn.upserts == []  # live chunk unchanged (same hash) → nothing re-embedded
    assert "STALE_KEY" in conn.deleted


def test_backfill_content_updates_only_rows_missing_content():
    from index.embed import _backfill_content

    units = [_unit("https://docs.pytorch.org/a.html"), _unit("https://docs.pytorch.org/b.html")]
    empty_key = chunk_key(units[0])  # DB reports only 'a' has empty content

    class Conn:
        def __init__(self):
            self.updates: list = []

        def execute(self, sql, params=None):
            assert "where content = ''" in sql  # only probes for un-backfilled rows
            return SimpleNamespace(fetchall=lambda: [(empty_key,)])

        def cursor(self):
            outer = self

            class Cur:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def executemany(self, sql, params):
                    outer.updates.extend(params)

            return Cur()

        def commit(self):
            pass

    conn = Conn()
    _backfill_content(conn, units)
    assert conn.updates == [("body text", empty_key)]  # only the empty row, with its text


def test_build_index_refuses_empty_snapshot(monkeypatch):
    # a cache-miss --skip-crawl (empty snapshot) must abort BEFORE the purge —
    # otherwise every db row looks stale and the whole live index gets wiped
    monkeypatch.setattr(embed, "iter_corpus_units", lambda corpus_dir=None: iter([]))
    with pytest.raises(SystemExit):
        build_index("v1", embed_fn=lambda texts: [])
