"""Day 3 benchmark on the product corpus: known-item search over Simple English Wikipedia.

    python bench/bench_wiki_rank.py --index data/index-en --n 500

Wikipedia has no relevance judgments, so this uses a known-item task: query = an article's title
(sampled, 2+ words), the one right answer = that article. Compares BM25 ranking with Day 2's
unranked boolean results (all matches in index order) and reports latency.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from seekr.index.query import Searcher
from seekr.index.rank import BM25, ranked_search
from seekr.index.reader import IndexReader


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index-en")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    reader = IndexReader(args.index)
    ranker = BM25(reader, k1=1.2, b=0.75)  # Seekr's product defaults (docs/day03-bm25.md)
    searcher = Searcher(reader)
    rng = random.Random(args.seed)
    picked: list[tuple[int, str]] = []
    while len(picked) < args.n:
        doc_id = rng.randrange(reader.n_docs)
        title = reader.db.execute("SELECT title FROM docs WHERE id = ?", (doc_id,)).fetchone()[0]
        if len(title.split()) >= 2 and len(reader.analyzer.terms(title)) >= 2:
            picked.append((doc_id, title))

    out = {}
    for mode in ("bm25", "unranked (Day 2)"):
        rr, s1, s10, times = [], 0, 0, []
        for doc_id, title in picked:
            t0 = time.perf_counter()
            if mode == "bm25":
                ranked = [d for d, _ in ranked_search(reader, title, k=10, ranker=ranker)]
            else:
                # Day 2 semantics: every document containing all the words, in index order.
                ranked = searcher.search(title).tolist()[:10]
            times.append((time.perf_counter() - t0) * 1000)
            rank = ranked.index(doc_id) + 1 if doc_id in ranked else None
            rr.append(1 / rank if rank else 0.0)
            s1 += rank == 1
            s10 += rank is not None
        out[mode] = {
            "queries": len(picked),
            "MRR@10": round(sum(rr) / len(rr), 4),
            "success@1": round(s1 / len(picked), 4),
            "success@10": round(s10 / len(picked), 4),
            "p50_ms": round(pct(times, 50), 1),
            "p95_ms": round(pct(times, 95), 1),
            "p99_ms": round(pct(times, 99), 1),
        }
        print(mode, out[mode], flush=True)
    examples = []
    for _, title in picked[:3]:
        top = ranked_search(reader, title, k=3, ranker=ranker)
        examples.append({"query": title, "top3": [reader.doc(d)["title"] for d, _ in top]})
    out["examples"] = examples
    print(json.dumps(examples, indent=2))
    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    (res / "day03_wiki_known_item.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
