"""Day 4: PageRank over Simple English Wikipedia, validated against Wikipedia's own signals.

    python bench/bench_pagerank.py --dump data/wiki/simplewiki-...json.gz --index data/index-en

Builds the link graph (aligned with the index's doc ids), computes PageRank and in-degree, and asks:
which one better predicts what readers actually look at (Wikimedia's pageview-based
`popularity_score`)? Saves links.npz and pagerank.npy into the index directory for ranking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from seekr.index.reader import IndexReader
from seekr.links.graph import build_link_graph, normalize_title
from seekr.links.pagerank import in_degree, pagerank
from seekr.links.stats import spearman, top_k_overlap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--index", default="data/index-en")
    args = ap.parse_args()
    index = Path(args.index)

    t = time.perf_counter()
    g = build_link_graph(args.dump)
    t_graph = time.perf_counter() - t
    print("graph", g.stats, f"{t_graph:.0f}s", flush=True)

    reader = IndexReader(index)
    index_titles = [normalize_title(x) for (x,) in reader.db.execute("SELECT title FROM docs ORDER BY id")]
    reader.close()
    aligned = index_titles == g.titles
    print("doc ids aligned with index:", aligned, flush=True)
    if not aligned:
        raise SystemExit("graph doc ids do not match the index; refusing to save")

    t = time.perf_counter()
    pr = pagerank(g.src, g.dst, g.n)
    t_pr = time.perf_counter() - t
    indeg = in_degree(g.dst, g.n)
    print(f"pagerank: {pr.iterations} iterations, converged={pr.converged}, {t_pr:.1f}s", flush=True)

    g.save(index / "links.npz")
    np.save(index / "pagerank.npy", pr.rank)

    has_views = g.popularity > 0
    corr = {
        "pagerank_vs_pageviews": spearman(pr.rank[has_views], g.popularity[has_views]),
        "indegree_vs_pageviews": spearman(indeg[has_views], g.popularity[has_views]),
        "wiki_incoming_vs_pageviews": spearman(g.incoming_links[has_views], g.popularity[has_views]),
        "pagerank_vs_indegree": spearman(pr.rank, indeg),
        "indegree_vs_wiki_incoming": spearman(indeg, g.incoming_links),
    }
    overlap = {
        f"top{k}_pagerank_vs_pageviews": top_k_overlap(pr.rank, g.popularity, k) for k in (100, 1000)
    } | {
        f"top{k}_indegree_vs_pageviews": top_k_overlap(indeg.astype(float), g.popularity, k)
        for k in (100, 1000)
    }
    top_pr = [g.titles[i] for i in np.argsort(-pr.rank)[:20]]
    top_in = [g.titles[i] for i in np.argsort(-indeg, kind="stable")[:20]]
    top_views = [g.titles[i] for i in np.argsort(-g.popularity)[:20]]
    outdeg = np.bincount(g.src, minlength=g.n)
    result = {
        "graph": g.stats | {"build_seconds": round(t_graph, 1), "dangling_pages": int((outdeg == 0).sum())},
        "pagerank": {
            "iterations": pr.iterations,
            "converged": pr.converged,
            "seconds": round(t_pr, 2),
            "ms_per_iteration": round(1000 * t_pr / pr.iterations, 1),
            "l1_history": [float(f"{x:.3e}") for x in pr.l1_history],
        },
        "pages_with_pageviews": int(has_views.sum()),
        "spearman": {k: round(v, 4) for k, v in corr.items()},
        "top_k_overlap": {k: round(v, 3) for k, v in overlap.items()},
        "top20_pagerank": top_pr,
        "top20_indegree": top_in,
        "top20_pageviews": top_views,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "pagerank"}, indent=2), flush=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "day04_pagerank.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
