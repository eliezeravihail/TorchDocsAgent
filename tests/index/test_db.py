"""index/db.py — meta helpers and the dimension-change rebuild, with a fake conn.

The DB is a rebuildable cache; the one piece of real logic worth pinning is
ensure_schema dropping and recreating chunks when the embedding dimension
changed (model swap). No live Postgres needed — a fake connection records the
SQL it was asked to run.
"""

from index.db import EMBED_DIMS, ensure_schema, get_meta, set_meta


class FakeCursor:
    def __init__(self, conn, sql, params):
        self.conn, self.sql, self.params = conn, sql, params

    def fetchone(self):
        if "atttypmod" in self.sql:
            return (self.conn.atttypmod,) if self.conn.atttypmod is not None else None
        if "select value from index_meta" in self.sql:
            key = self.params[0]
            return (self.conn.meta[key],) if key in self.conn.meta else None
        return None


class FakeConn:
    def __init__(self, atttypmod=None, meta=None):
        self.atttypmod = atttypmod
        self.meta = dict(meta or {})
        self.executed: list[str] = []
        self.committed = False

    def execute(self, sql, params=None):
        self.executed.append(sql)
        return FakeCursor(self, sql, params)

    def commit(self):
        self.committed = True


def _dropped(conn: FakeConn) -> bool:
    return any("drop table chunks" in sql for sql in conn.executed)


def test_get_meta_returns_value_or_none():
    conn = FakeConn(meta={"embed_recipe": "v3"})
    assert get_meta(conn, "embed_recipe") == "v3"
    assert get_meta(conn, "missing") is None


def test_set_meta_issues_upsert():
    conn = FakeConn()
    set_meta(conn, "embed_recipe", "v4")
    assert any("insert into index_meta" in sql and "on conflict" in sql for sql in conn.executed)


def test_ensure_schema_rebuilds_on_dimension_change():
    conn = FakeConn(atttypmod=128)  # a different embedding width than EMBED_DIMS
    ensure_schema(conn)
    assert _dropped(conn)  # old vector space dropped
    assert conn.committed


def test_ensure_schema_keeps_table_when_dimension_matches():
    conn = FakeConn(atttypmod=EMBED_DIMS)
    ensure_schema(conn)
    assert not _dropped(conn)
    assert conn.committed


def test_ensure_schema_fresh_table_no_drop():
    conn = FakeConn(atttypmod=None)  # column introspection returns nothing
    ensure_schema(conn)
    assert not _dropped(conn)
