"""Day 1 benchmark: throughput, politeness and dedup accuracy on a synthetic web with ground truth.

    python bench/bench_crawler.py            # writes bench/results/day01.json and prints a table

Throughput is bounded by politeness, not by the network: with H hosts, delay d and latency l,
the ceiling is H / (d + l) pages/s. The number to beat is how close the scheduler gets to it.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from seekr.crawl.crawler import CrawlConfig, Crawler
from seekr.crawl.simweb import SimWeb
from seekr.crawl.store import PageStore

HOSTS, PAGES, LATENCY, DELAY = 20, 100, 0.05, 0.25


def score(web: SimWeb, store: PageStore, log: list[tuple[str, float]]) -> dict:
    rows = store.conn.execute("SELECT url, dup_of FROM pages WHERE status = 200").fetchall()
    members = defaultdict(list)
    for url, _ in rows:
        members[web.cluster(url)].append(url)
    flagged = [(u, d) for u, d in rows if d]
    correct = sum(web.cluster(u) == web.cluster(d) for u, d in flagged)
    should = sum(len(m) - 1 for m in members.values())
    times = defaultdict(list)
    for url, t in log:
        times[url.split("/")[2]].append(t)
    gaps = [b - a for ts in times.values() for a, b in zip(sorted(ts), sorted(ts)[1:], strict=False)]
    return {
        "dup_precision": round(correct / len(flagged), 4) if flagged else 1.0,
        "dup_recall": round(correct / should, 4) if should else 1.0,
        "robots_violations": sum("/private/" in u for u, _ in log),
        "tracking_param_refetches": sum("utm_" in u for u, _ in log),
        "min_gap_same_host_s": round(min(gaps), 4),
        "politeness_violations": sum(g < DELAY - 1e-6 for g in gaps),
    }


async def run_one(concurrency: int, cpu_workers: int) -> dict:
    web = SimWeb(n_hosts=HOSTS, pages_per_host=PAGES, latency=LATENCY, seed=21)
    log: list[tuple[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = PageStore(Path(tmp) / "bench.db")
        cfg = CrawlConfig(
            seeds=[f"http://{h}/page/0" for h in web.hosts],
            max_pages=100_000,
            max_depth=50,
            concurrency=concurrency,
            delay=DELAY,
            cpu_workers=cpu_workers,
        )
        stats = await Crawler(cfg, store, web.transport(), on_request=lambda u, t: log.append((u, t))).run()
        result = {
            "concurrency": concurrency,
            "cpu_workers": cpu_workers,
            **stats.as_dict(),
            **score(web, store, log),
        }
        store.close()
    result["ceiling_pages_per_sec"] = round(HOSTS / (DELAY + LATENCY), 1)
    result["pct_of_ceiling"] = round(100 * result["pages_per_sec"] / result["ceiling_pages_per_sec"], 1)
    return result


def main() -> None:
    results = [asyncio.run(run_one(c, w)) for c, w in ((1, 0), (32, 0), (32, 4))]
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "day01.json").write_text(json.dumps(results, indent=2))
    cols = [
        "concurrency",
        "cpu_workers",
        "fetched",
        "pages_per_sec",
        "pct_of_ceiling",
        "dup_precision",
        "dup_recall",
        "exact_dups",
        "near_dups",
        "near_dup_rejected",
        "robots_blocked",
        "robots_violations",
        "politeness_violations",
        "min_gap_same_host_s",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
