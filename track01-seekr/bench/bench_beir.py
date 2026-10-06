"""Day 3 benchmark: from-scratch BM25 vs Anserini's published BM25 on five BEIR datasets.

    python bench/bench_beir.py --data data/beir --workers 3

Configurations isolate one factor each:
  A  plain tokens                          k1=0.9 b=0.4  Lucene norms
  B  english analyzer, exact doc lengths   k1=0.9 b=0.4
  C  english analyzer, Lucene norms        k1=0.9 b=0.4   <- Anserini's setup
  D  english analyzer, Lucene norms        k1=1.2 b=0.75  (textbook defaults)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from seekr.eval import beir

DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa", "trec-covid"]
CONFIGS = {
    "A plain": ("plain", 0.9, 0.4, True),
    "B english, exact lengths": ("english", 0.9, 0.4, False),
    "C english, Lucene norms": ("english", 0.9, 0.4, True),
    "D english, k1=1.2 b=0.75": ("english", 1.2, 0.75, True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/beir")
    ap.add_argument("--indexes", default="data/beir-index")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()

    results = []
    for name in args.datasets:
        d = Path(args.data) / name
        built = {}
        for analyzer in ("plain", "english"):
            t = time.perf_counter()
            stats = beir.build(d, Path(args.indexes) / f"{name}-{analyzer}", analyzer, workers=args.workers)
            built[analyzer] = {
                "docs": stats["docs"],
                "terms": stats["terms"],
                "build_s": round(time.perf_counter() - t, 1),
            }
            print(f"{name}: built {analyzer} index {built[analyzer]}", flush=True)
        ref = beir.ANSERINI_BM25_FLAT[name]
        for label, (analyzer, k1, b, norms) in CONFIGS.items():
            run, timing = beir.run_bm25(
                d, Path(args.indexes) / f"{name}-{analyzer}", k1=k1, b=b, lucene_norms=norms
            )
            m = beir.evaluate_dataset(d, run)
            row = {
                "dataset": name,
                "config": label,
                "nDCG@10": round(m["nDCG@10"], 4),
                "R@100": round(m["R@100"], 4),
                "R@1000": round(m["R@1000"], 4),
                "anserini_nDCG@10": ref[0],
                "delta_nDCG@10": round(m["nDCG@10"] - ref[0], 4),
                "queries": m["queries"],
                "p50_ms": round(timing["p50_ms"], 1),
                "p99_ms": round(timing["p99_ms"], 1),
                **{f"{k}_{analyzer}": v for k, v in built[analyzer].items()},
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "day03_beir.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
