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


def test_get_pool_applies_runtime_migrations(monkeypatch):
    # a deploy can SELECT a new column (e.g. `part`) before any index build
    # runs ensure_schema — the pool itself must bring the table up to date
    import sys
    import types

    from index import db

    executed = []

    class FakeConn:
        def execute(self, sql, *args):
            executed.append(sql)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakePool:
        check_connection = staticmethod(lambda conn: None)

        def __init__(self, url, **kwargs):
            pass

        def open(self):
            pass

        def connection(self):
            return FakeConn()

    monkeypatch.setenv("NEON_URL", "postgresql://user:pw@host/db")
    monkeypatch.setitem(
        sys.modules, "psycopg_pool", types.SimpleNamespace(ConnectionPool=FakePool)
    )
    db.get_pool.cache_clear()
    try:
        db.get_pool()
    finally:
        db.get_pool.cache_clear()
    assert any("add column if not exists part" in sql for sql in executed)


def test_ensure_schema_applies_runtime_migrations():
    from index.db import RUNTIME_MIGRATIONS, SCHEMA

    # the migration list is the single source both writers consume; the base
    # SCHEMA must already contain each migrated column for fresh tables
    assert any("part" in m for m in RUNTIME_MIGRATIONS)
    assert "part" in SCHEMA


def test_embed_dims_derives_from_the_model(monkeypatch):
    from index.db import embed_dims

    # dims track the model, so db.py and index/embed.py can't drift apart
    monkeypatch.delenv("TORCHDOCS_EMBED_DIMS", raising=False)
    monkeypatch.setenv("TORCHDOCS_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    assert embed_dims() == 768
    monkeypatch.setenv("TORCHDOCS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    assert embed_dims() == 384


def test_embed_dims_unknown_model_needs_an_override(monkeypatch):
    import pytest

    from index.db import embed_dims

    monkeypatch.delenv("TORCHDOCS_EMBED_DIMS", raising=False)
    monkeypatch.setenv("TORCHDOCS_EMBED_MODEL", "some/unknown-embedder")
    # an unknown model with no override is a config error, not a silent
    # wrong-width table
    with pytest.raises(RuntimeError, match="unknown embed model"):
        embed_dims()
    monkeypatch.setenv("TORCHDOCS_EMBED_DIMS", "512")
    assert embed_dims() == 512
