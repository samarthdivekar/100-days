"""Day 2 benchmark: index build throughput, inline vs parallel, on the first N Wikipedia articles.

python bench/bench_index_build.py --dump data/wiki/simplewiki-...json.gz --docs 8000 --workers 0 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from seekr.index.builder import IndexBuilder
from seekr.index.corpus import from_cirrus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--docs", type=int, default=8000)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--block-tokens", type=int, default=1_000_000)
    args = ap.parse_args()

    docs = list(from_cirrus(args.dump, limit=args.docs))
    results = []
    for w in args.workers:
        out = Path(f"data/bench-build-w{w}")
        t = time.perf_counter()
        builder = IndexBuilder(out, block_tokens=args.block_tokens, workers=w)
        builder.add_all(docs)
        stats = builder.finish()
        total = time.perf_counter() - t
        results.append(
            {
                "workers": w,
                "docs": stats["docs"],
                "tokens": stats["tokens"],
                "seconds": round(total, 1),
                "merge_seconds": stats["merge_seconds"],
                "tokens_per_sec": round(stats["tokens"] / total),
            }
        )
        print(results[-1])
    base = results[0]["seconds"]
    ref = (Path(f"data/bench-build-w{args.workers[0]}") / "postings.bin").read_bytes()
    for r in results:
        r["speedup"] = round(base / r["seconds"], 2)
        r["byte_identical"] = (Path(f"data/bench-build-w{r['workers']}") / "postings.bin").read_bytes() == ref
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "day02_build.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
