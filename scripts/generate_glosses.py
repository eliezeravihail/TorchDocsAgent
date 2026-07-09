"""Generate search glosses for API reference pages (Contextual Retrieval).

Why: the retrieval diagnosis showed descriptive questions ("which loss takes
raw logits and a target class index?") never surface the terse core reference
pages (CrossEntropyLoss, SGD, LayerNorm) — the pages' indexed text is
signature/parameter-shaped, so it embeds far from question vocabulary. The
standard fix (Anthropic's Contextual Retrieval) is to prepend a short
plain-language context line to each chunk before embedding; it cut retrieval
failures by ~49% on their benchmark, and it improves BOTH channels here since
indexed_text() also feeds the tsvector.

What: for every api-kind page in the corpus snapshot, ask an LLM for a 1-2
sentence gloss — what it is, what a user is trying to do when they need it,
in everyday ML vocabulary ("fully-connected layer" for Linear). Batched
(BATCH pages per call) so the 3.6K-page corpus fits in a few hundred calls;
resumable (URLs already glossed are skipped, output is appended and flushed
per batch) so rate-limit deaths just mean "run it again".

Output: index/glosses.jsonl — {"url", "gloss"} per line, committed, folded
into indexed_text() and the embed recipe by index/embed.py.

Usage:  python scripts/generate_glosses.py [--limit N] [--batch N]
        (needs an LLM key; corpus snapshot must exist — run the crawl first)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

GLOSSES_PATH = Path(__file__).parent.parent / "index" / "glosses.jsonl"

SYSTEM = (
    "You write search glosses for PyTorch documentation reference pages. For "
    "each numbered page (symbol/title + excerpt) write ONE sentence, 15-35 "
    "words, in plain English: what it is/computes and what a user is trying "
    "to do when they need it. Use everyday ML vocabulary and likely "
    "paraphrases a user would search with (e.g. 'fully-connected layer' for "
    "Linear, 'multi-class classification loss on raw logits' for "
    "CrossEntropyLoss) — do not just restate the symbol name. Reply with a "
    "JSON array only, one item per page, no other text: "
    '[{"i": 0, "gloss": "..."}, ...]'
)

EXCERPT_CHARS = 700  # enough for signature + the first description sentence
GLOSS_MAX_CHARS = 350


def api_pages(corpus_dir: Path) -> list[dict]:
    """Every api-kind page in the snapshot: {url, title, excerpt}. Core first."""
    from ingest.chunk_docs import page_kind
    from ingest.crawl import load_page

    pages = []
    for path in sorted(corpus_dir.rglob("*.md")):
        meta, body = load_page(path)
        if page_kind(meta["url"]) != "api":
            continue
        pages.append(
            {
                "url": meta["url"],
                "title": meta.get("title", ""),
                "excerpt": re.sub(r"\s+", " ", body[:EXCERPT_CHARS]).strip(),
            }
        )
    # the measured misses are all core-torch pages — gloss those first so a
    # partial (rate-limited) run still covers the pages that matter most
    pages.sort(key=lambda p: (0 if "/docs/stable/" in p["url"] else 1, p["url"]))
    return pages


def batch_prompt(batch: list[dict]) -> str:
    from index.embed import symbol_from_url

    blocks = []
    for i, page in enumerate(batch):
        symbol = symbol_from_url(page["url"]) or page["title"]
        blocks.append(f"### {i}\nsymbol: {symbol}\nexcerpt: {page['excerpt']}")
    return "\n\n".join(blocks) + f"\n\nJSON array with {len(batch)} glosses:"


def parse_glosses(raw: str, n: int) -> dict[int, str]:
    """{index: gloss} from the model's reply; malformed items are dropped."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return {}
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[int, str] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        i, gloss = item.get("i"), item.get("gloss")
        if isinstance(i, int) and 0 <= i < n and isinstance(gloss, str) and gloss.strip():
            out[i] = gloss.strip()[:GLOSS_MAX_CHARS]
    return out


def existing_urls_of(path: Path) -> set[str]:
    """URLs already covered in a jsonl enrichment file — the resume check,
    shared with generate_questions.py (same append-and-skip pipeline shape)."""
    if not path.exists():
        return set()
    return {json.loads(line)["url"] for line in path.open() if line.strip()}


def existing_urls() -> set[str]:
    return existing_urls_of(GLOSSES_PATH)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="gloss at most N pages (0 = all)")
    parser.add_argument("--batch", type=int, default=12, help="pages per LLM call")
    parser.add_argument("--sleep", type=float, default=2.0, help="pause between calls (s)")
    args = parser.parse_args()

    from agent.llm import GenerationError, _raw_completion
    from ingest.crawl import CORPUS_DIR

    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.rglob("*.md")):
        print("corpus snapshot is empty — run the crawl (Build Index) first", flush=True)
        return 1

    done = existing_urls()
    todo = [p for p in api_pages(CORPUS_DIR) if p["url"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[gloss] {len(done)} already glossed, {len(todo)} to go", flush=True)
    if not todo:
        return 0

    written = failed_batches = 0
    with GLOSSES_PATH.open("a") as out:
        for at in range(0, len(todo), args.batch):
            batch = todo[at : at + args.batch]
            try:
                raw = _raw_completion(batch_prompt(batch), system=SYSTEM, timeout=120.0)
            except GenerationError as exc:
                print(f"[gloss] batch at {at} failed: {exc}", flush=True)
                failed_batches += 1
                if failed_batches >= 5:
                    print("[gloss] 5 failed batches — provider looks down, stopping", flush=True)
                    break
                continue
            glosses = parse_glosses(raw, len(batch))
            if not glosses:
                print(f"[gloss] batch at {at}: unparseable reply, skipped", flush=True)
                failed_batches += 1
                continue
            for i, gloss in sorted(glosses.items()):
                out.write(json.dumps({"url": batch[i]["url"], "gloss": gloss},
                                     ensure_ascii=False) + "\n")
            out.flush()  # checkpoint: kill/rate-limit here loses nothing
            written += len(glosses)
            print(f"[gloss] {at + len(batch)}/{len(todo)} pages seen, "
                  f"{written} glosses written", flush=True)
            time.sleep(args.sleep)

    print(f"[gloss] done: {written} new glosses → {GLOSSES_PATH}", flush=True)
    # partial success is success (resumable); total failure is loud
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
