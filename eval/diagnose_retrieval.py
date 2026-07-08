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
PROBES = [
    ("a06", "What's the standard fully-connected layer that applies a weight matrix "
            "and bias?", "torch.nn.linear"),
    ("a17", "For multi-class classification, which loss takes raw logits and the "
            "target class index?", "crossentropyloss"),
    ("c02", "What exactly is SGD's parameter update step, including momentum, "
            "mathematically?", "torch.optim.sgd"),
    ("a10", "Which normalization layer works per-sample across features, the one "
            "transformers use?", "layernorm"),
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
        from index.retrieve import HNSW_EF_SEARCH

        # mirror retrieve(): widen the approximate scan so kind-filtered dense
        # queries here behave like the product path
        conn.execute(f"set hnsw.ef_search = {HNSW_EF_SEARCH:d}")
        for qid, question, expected in PROBES:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
