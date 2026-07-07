"""Smoke test: verify every external connection works before building anything.

Checks, in order:
  1. Neon Postgres    — connect, create a scratch table, write one row, read it back.
  2. Gemini LLM       — one short message round-trip.
  3. Embeddings       — one local bge-small call, sanity-check the vector dimension.
  4. Anthropic LLM    — one short message round-trip (optional; skipped if no key).

Run:  python scripts/smoke.py
Exits 0 only if every configured check passes. A missing key skips that
check (Anthropic) or fails it with a clear message instead of a traceback,
so partial setups still give useful output.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

PASS = "✓"
FAIL = "✗"
SKIP = "__skip__"  # sentinel: check not configured and not required


def check_neon() -> str | None:
    url = os.environ.get("NEON_URL")
    if not url:
        return "NEON_URL is not set (copy .env.example to .env and fill it in)"
    import psycopg

    with psycopg.connect(url, connect_timeout=15) as conn, conn.cursor() as cur:
        cur.execute("create table if not exists smoke (id int primary key, note text)")
        cur.execute(
            "insert into smoke (id, note) values (1, 'hello from smoke.py') "
            "on conflict (id) do update set note = excluded.note"
        )
        cur.execute("select note from smoke where id = 1")
        row = cur.fetchone()
        if row is None or "hello" not in row[0]:
            return f"read-back mismatch: {row!r}"
        cur.execute("drop table smoke")
    return None


def check_gemini_llm() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "GEMINI_API_KEY is not set (get one free at aistudio.google.com)"
    from google import genai

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with the single word: pong",
    )
    if "pong" not in (response.text or "").lower():
        return f"unexpected reply: {response.text!r}"
    return None


def check_anthropic_llm() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return SKIP  # optional — Anthropic is the paid production provider
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=60)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")
    if "pong" not in text.lower():
        return f"unexpected reply: {text!r}"
    return None


def check_embedding() -> str | None:
    from index.db import EMBED_DIMS
    from index.embed import embed_texts

    vector = embed_texts(["torch.nn.Linear applies an affine transformation"])[0]
    if len(vector) != EMBED_DIMS or all(v == 0 for v in vector):
        return f"suspicious embedding: len={len(vector)} (expected {EMBED_DIMS})"
    return None


def main() -> int:
    load_dotenv()
    checks = [
        ("Neon Postgres (write/read)", check_neon),
        ("Gemini LLM (one message)", check_gemini_llm),
        ("Local embedding (bge-small, one vector)", check_embedding),
        ("Anthropic LLM (optional)", check_anthropic_llm),
    ]
    failures = 0
    skipped = 0
    for name, fn in checks:
        try:
            error = fn()
        except Exception as exc:  # a broken connection should report, not crash the run
            error = f"{type(exc).__name__}: {exc}"
        if error == SKIP:
            skipped += 1
            print(f"- {name}: skipped (no key)")
        elif error:
            failures += 1
            print(f"{FAIL} {name}: {error}")
        else:
            print(f"{PASS} {name}")
    required = len(checks) - skipped
    print(f"\n{required - failures}/{required} required checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
