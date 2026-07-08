"""Why does a descriptive question miss its API reference page?

Not a benchmark — a microscope. For a handful of known misses it prints, per
kind-pool, the nearest candidates WITH cosine distances, and separately locates
the EXPECTED page in the raw dense/keyword candidates so we can see whether it
was (a) never a candidate, (b) a candidate but out-ranked inside its own pool,
or (c) a candidate that the relevance-gap filter dropped. That triage picks the
fix. Run in Actions (needs NEON_URL). One-off; delete once the fix lands.

Usage:  python -m eval.diagnose_retrieval
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

# (qid, question, expected-substring) — a representative slice of the API/code
# misses from retrieval_v1: all descriptive, none containing the symbol token.
# Each probe also carries a `lemma` from the page's own docstring first sentence
# (the synopsis line) that is NOT present in its signature/params — used by the
# tsv reality check below to confirm the descriptive prose actually reached the
# indexed text (and so whether a byte-identical distance is a real negative or a
# stale index that never got the enrichment).
PROBES = [
    ("a06", "What's the standard fully-connected layer that applies a weight matrix "
            "and bias?", "torch.nn.linear", "transformation"),
    ("a17", "For multi-class classification, which loss takes raw logits and the "
            "target class index?", "crossentropyloss", "criterion"),
    ("c02", "What exactly is SGD's parameter update step, including momentum, "
            "mathematically?", "torch.optim.sgd", "descent"),
    ("a10", "Which normalization layer works per-sample across features, the one "
            "transformers use?", "layernorm", "normalization"),
]

POOL = 20


def main() -> int:
    load_dotenv()
    from index.db import get_pool
    from index.embed import embed_query
    from index.retrieve import (
        DENSE_SQL,
        KEYWORD_SQL,
        KINDS,
        POINTER_COLUMNS,
    )

    ncols = len(POINTER_COLUMNS.replace(" ", "").replace("\n", "").split(","))
    url_i = POINTER_COLUMNS.replace(" ", "").replace("\n", "").split(",").index("url")

    with get_pool().connection() as conn:
        from index.db import get_meta
        from index.embed import EMBED_RECIPE
        from index.retrieve import HNSW_EF_SEARCH

        # Recipe-liveness gate. Every distance below is only meaningful against
        # the index the CODE describes. If the live index was built under an
        # older recipe, a "byte-identical distance" is not a failed enrichment —
        # it is a stale index that never got the enrichment. Print both and say
        # so loudly, so we never again read a stale number as a real negative.
        live_recipe = get_meta(conn, "embed_recipe")
        match = "MATCH" if live_recipe == EMBED_RECIPE else "STALE — REBUILD BEFORE TRUSTING"
        print(f"recipe: code={EMBED_RECIPE!r}  live={live_recipe!r}  [{match}]")

        # mirror retrieve(): widen the approximate scan so kind-filtered dense
        # queries here behave like the product path
        conn.execute(f"set hnsw.ef_search = {HNSW_EF_SEARCH:d}")
        for qid, question, expected, lemma in PROBES:
            print(f"\n{'=' * 78}\n{qid}: {question}\n  expected page contains: {expected!r}")
            vec = str(embed_query(question))
            for kd in KINDS:
                params = {"pool": POOL, "query": question, "vector": vec, "kind": kd}
                where = "where kind = %(kind)s"
                extra = "and kind = %(kind)s "
                dense = conn.execute(DENSE_SQL.format(where=where), params).fetchall()
                kw = conn.execute(KEYWORD_SQL.format(extra=extra), params).fetchall()
                print(f"\n  [{kd}] dense top-3 (dist):")
                for row in dense[:3]:
                    print(f"      {row[ncols]:.3f}  {row[url_i]}")
                # where does the expected page sit in this kind's dense list?
                d_rank = next(
                    (i for i, r in enumerate(dense, 1) if expected in r[url_i].lower()), None
                )
                d_dist = next(
                    (r[ncols] for r in dense if expected in r[url_i].lower()), None
                )
                k_rank = next(
                    (i for i, r in enumerate(kw, 1) if expected in r[url_i].lower()), None
                )
                loc = []
                if d_rank:
                    loc.append(f"dense#{d_rank} dist={d_dist:.3f}")
                if k_rank:
                    loc.append(f"keyword#{k_rank}")
                where_txt = ", ".join(loc) if loc else "ABSENT from top-20"
                print(f"      → expected in [{kd}]: {where_txt}")

            # The decisive number: the expected page's OWN nearest chunk — its
            # absolute cosine distance, and its true dense rank among ALL api
            # chunks (count of api chunks strictly closer). rank ~25 = a crowding
            # problem (a deeper pool / rerank helps); a far distance / rank in the
            # hundreds = an embedding problem (only doc-side enrichment helps).
            pat = f"%{expected}%"
            best = conn.execute(
                "select min(embedding <=> %(vector)s::vector) from chunks "
                "where kind = 'api' and url ilike %(pat)s",
                {"vector": vec, "pat": pat},
            ).fetchone()[0]
            if best is None:
                print(f"\n  >>> expected page {expected!r}: no api chunk with that url")
            else:
                closer = conn.execute(
                    "select count(*) from chunks where kind = 'api' "
                    "and (embedding <=> %(vector)s::vector) < %(best)s",
                    {"vector": vec, "best": best},
                ).fetchone()[0]
                print(
                    f"\n  >>> expected page {expected!r}: nearest api chunk dist="
                    f"{best:.3f}, true dense rank in api = {closer + 1} "
                    f"(api-pool cutoff is top-{POOL})"
                )

            # tsv reality check: is the page's own docstring prose (the synopsis
            # line + description body) actually in the indexed text? The lemma is
            # a descriptive word from the first sentence that does NOT appear in
            # the signature/params, so a match proves the prose reached the tsv
            # (and, since indexed_text() feeds one string to both, the vector).
            # No match on a MATCH recipe = the enrichment genuinely isn't landing
            # for this page — a real content problem, not a measurement artifact.
            hit = conn.execute(
                "select count(*) from chunks where kind = 'api' and url ilike %(pat)s "
                "and tsv @@ plainto_tsquery('english', %(lemma)s)",
                {"pat": pat, "lemma": lemma},
            ).fetchone()[0]
            total = conn.execute(
                "select count(*) from chunks where kind = 'api' and url ilike %(pat)s",
                {"pat": pat},
            ).fetchone()[0]
            print(
                f"  >>> descriptive prose check: {hit}/{total} api chunks of "
                f"{expected!r} contain lemma {lemma!r} "
                f"({'PROSE INDEXED' if hit else 'PROSE ABSENT — synopsis/body not landing'})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
