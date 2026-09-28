"""Day 2 benchmark: index size and query latency on the full index.

    python bench/bench_query.py --index data/index

Query sets are sampled from the indexed articles themselves (seeded), so every query has at least
one known answer: a phrase taken from article X must return X, which doubles as a correctness
check at full scale.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from seekr.index.query import Searcher
from seekr.index.reader import IndexReader
from seekr.index.tokenize import tokenize

WORST_CASE = ['"of the"', '"in the"', "the", '"one of the"', "the of and"]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def sample_queries(reader: IndexReader, n: int, seed: int) -> dict[str, list[tuple[str, int]]]:
    rng = random.Random(seed)
    sets: dict[str, list[tuple[str, int]]] = {"term": [], "and2": [], "phrase2": [], "phrase3": []}
    while min(len(v) for v in sets.values()) < n:
        doc_id = rng.randrange(reader.n_docs)
        d = reader.doc(doc_id)
        toks = tokenize(f"{d['title']}\n{d['text']}")
        if len(toks) < 10:
            continue
        i = rng.randrange(len(toks) - 3)
        if len(sets["term"]) < n:
            sets["term"].append((toks[i], doc_id))
        if len(sets["and2"]) < n:
            sets["and2"].append((f"{toks[i]} {rng.choice(toks)}", doc_id))
        if len(sets["phrase2"]) < n:
            sets["phrase2"].append((f'"{toks[i]} {toks[i + 1]}"', doc_id))
        if len(sets["phrase3"]) < n:
            sets["phrase3"].append((f'"{toks[i]} {toks[i + 1]} {toks[i + 2]}"', doc_id))
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    reader = IndexReader(args.index)
    searcher = Searcher(reader)
    meta = reader.meta
    index_dir = Path(args.index)
    size = {
        "docs": meta["docs"],
        "tokens": meta["tokens"],
        "terms": meta["terms"],
        "postings": meta["postings"],
        "raw_text_mb": round(meta["raw_text_bytes"] / 1e6, 1),
        "postings_mb": round(meta["postings_bytes"] / 1e6, 1),
        "docid_mb": round(meta["docid_bytes"] / 1e6, 1),
        "tf_mb": round(meta["tf_bytes"] / 1e6, 1),
        "position_mb": round(meta["position_bytes"] / 1e6, 1),
        "uncompressed_32bit_mb": round(meta["uncompressed_bytes"] / 1e6, 1),
        "compression_vs_32bit": round(meta["uncompressed_bytes"] / meta["postings_bytes"], 2),
        "postings_pct_of_raw_text": round(100 * meta["postings_bytes"] / meta["raw_text_bytes"], 1),
        "bits_per_position": round(8 * meta["position_bytes"] / meta["tokens"], 2),
        "bits_per_docid": round(8 * meta["docid_bytes"] / meta["postings"], 2),
        "index_db_mb": round((index_dir / "index.db").stat().st_size / 1e6, 1),
        "build_seconds": meta["build_seconds"],
        "merge_seconds": meta["merge_seconds"],
    }
    print(json.dumps(size, indent=2))

    searcher.search("warmup")
    sets = sample_queries(reader, args.n, args.seed)
    latency: dict[str, dict] = {}
    for kind, queries in sets.items():
        times, hits, misses = [], [], 0
        for q, source in queries:
            t0 = time.perf_counter()
            result = searcher.search(q)
            times.append((time.perf_counter() - t0) * 1000)
            hits.append(len(result))
            if kind.startswith("phrase") and source not in set(result.tolist()):
                misses += 1
        latency[kind] = {
            "queries": len(queries),
            "p50_ms": round(pct(times, 50), 2),
            "p95_ms": round(pct(times, 95), 2),
            "p99_ms": round(pct(times, 99), 2),
            "mean_ms": round(statistics.mean(times), 2),
            "median_results": int(statistics.median(hits)),
            "source_doc_missing": misses,
        }
        print(kind, latency[kind])
    worst = {}
    for q in WORST_CASE:
        t0 = time.perf_counter()
        n = len(searcher.search(q))
        worst[q] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "results": n}
    print("worst cases", worst)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "day02_query.json").write_text(
        json.dumps({"size": size, "latency": latency, "worst": worst}, indent=2)
    )


if __name__ == "__main__":
    main()
