"""Generate hypothetical questions for API reference pages (QuOTE-style).

Why: the reranker measurement (2026-07-09) left a residue of genuinely-buried
pages — Linear, Conv2d, random_split, WeightedRandomSampler, einsum — where a
descriptive question ("what's the standard fully-connected layer?") matches
NOTHING the page carries, not even its one-sentence gloss (Linear still sat at
true dense rank ~3,400 post-gloss). The literature's index-side answer
(QuOTE, arXiv:2502.10976; HyPE) is to index the QUESTIONS a page answers,
turning question→document matching into question→question matching — paid
once at index time, not per query like HyDE.

What: for every api-kind page in the corpus snapshot, ask an LLM for a few
short questions a user would ask that this page answers — phrased in everyday
task vocabulary, mostly WITHOUT naming the symbol (the vocabulary bridge is
the whole point; the symbol token is already in the index). The questions are
folded into indexed_text() by index/embed.py — feeding both the page's vector
and its tsvector, the exact channel pair that flipped CrossEntropyLoss.

Output: index/questions.jsonl — {"url", "questions": [...]} per line,
committed. Same shape as the gloss pipeline: batched, flushed per batch,
resumable (already-covered URLs are skipped) — rate-limit deaths just mean
"run it again".

Usage:  python scripts/generate_questions.py [--limit N] [--batch N]
        (needs an LLM key; corpus snapshot must exist — run the crawl first)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from scripts.generate_glosses import api_pages, existing_urls_of

QUESTIONS_PATH = Path(__file__).parent.parent / "index" / "questions.jsonl"

QUESTIONS_PER_PAGE = 5
QUESTION_MAX_CHARS = 160

SYSTEM = (
    "You write search-bridging questions for PyTorch documentation reference "
    f"pages. For each numbered page (symbol/title + excerpt) write "
    f"{QUESTIONS_PER_PAGE} distinct short questions (8-20 words) a PyTorch "
    "user would ask that THIS page answers. Phrase them the way users talk "
    "about the TASK, in everyday ML vocabulary; at most one question may name "
    "the symbol itself — the rest must describe what it does or when you need "
    "it (e.g. for Linear: 'What's the standard fully-connected layer that "
    "applies a weight matrix and bias?'). Reply with a JSON array only, one "
    'item per page, no other text: [{"i": 0, "questions": ["...", ...]}, ...]'
)


def batch_prompt(batch: list[dict]) -> str:
    from index.embed import symbol_from_url

    blocks = []
    for i, page in enumerate(batch):
        symbol = symbol_from_url(page["url"]) or page["title"]
        blocks.append(f"### {i}\nsymbol: {symbol}\nexcerpt: {page['excerpt']}")
    return "\n\n".join(blocks) + f"\n\nJSON array with {len(batch)} question sets:"


def parse_questions(raw: str, n: int) -> dict[int, list[str]]:
    """{index: [questions]} from the model's reply; malformed items are dropped."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return {}
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[int, list[str]] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        i, qs = item.get("i"), item.get("questions")
        if not (isinstance(i, int) and 0 <= i < n and isinstance(qs, list)):
            continue
        clean = [q.strip()[:QUESTION_MAX_CHARS] for q in qs if isinstance(q, str) and q.strip()]
        if clean:
            out[i] = clean[:QUESTIONS_PER_PAGE]
    return out


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="cover at most N pages (0 = all)")
    parser.add_argument("--batch", type=int, default=10, help="pages per LLM call")
    parser.add_argument("--sleep", type=float, default=2.0, help="pause between calls (s)")
    args = parser.parse_args()

    from agent.llm import GenerationError, _raw_completion
    from ingest.crawl import CORPUS_DIR

    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.rglob("*.md")):
        print("corpus snapshot is empty — run the crawl (Build Index) first", flush=True)
        return 1

    done = existing_urls_of(QUESTIONS_PATH)
    todo = [p for p in api_pages(CORPUS_DIR) if p["url"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[questions] {len(done)} pages already covered, {len(todo)} to go", flush=True)
    if not todo:
        return 0

    written = failed_batches = 0
    with QUESTIONS_PATH.open("a") as out:
        for at in range(0, len(todo), args.batch):
            batch = todo[at : at + args.batch]
            try:
                raw = _raw_completion(batch_prompt(batch), system=SYSTEM, timeout=120.0)
            except GenerationError as exc:
                print(f"[questions] batch at {at} failed: {exc}", flush=True)
                failed_batches += 1
                if failed_batches >= 5:
                    print("[questions] 5 failed batches — provider looks down, stopping",
                          flush=True)
                    break
                continue
            sets = parse_questions(raw, len(batch))
            if not sets:
                print(f"[questions] batch at {at}: unparseable reply, skipped", flush=True)
                failed_batches += 1
                continue
            for i, qs in sorted(sets.items()):
                out.write(json.dumps({"url": batch[i]["url"], "questions": qs},
                                     ensure_ascii=False) + "\n")
            out.flush()  # checkpoint: kill/rate-limit here loses nothing
            written += len(sets)
            print(f"[questions] {at + len(batch)}/{len(todo)} pages seen, "
                  f"{written} question sets written", flush=True)
            time.sleep(args.sleep)

    print(f"[questions] done: {written} new question sets → {QUESTIONS_PATH}", flush=True)
    # partial success is success (resumable); total failure is loud
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
