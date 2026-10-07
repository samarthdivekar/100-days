"""Day 4: does a link-based prior (PageRank, or plain in-degree) make Seekr's search better?

    python bench/bench_prior.py --index data/index-en

Known-item queries (query = an article's title, the article is the answer), two kinds:
  * traffic : articles sampled in proportion to their pageviews (what people actually look up)
  * uniform : every article equally likely (the worst case for a popularity prior)

The prior weight w is tuned on a TRAIN traffic set and reported on disjoint TEST sets, so the
reported gains are not chosen by looking at the test queries. Final score = BM25 + w * prior.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from seekr.index.query import parse
from seekr.index.rank import BM25, load_prior
from seekr.index.reader import IndexReader

WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def sample(rng, n, probs, exclude, reader):
    out, seen = [], set(exclude)
    while len(out) < n:
        doc = (
            int(rng.choice(probs.size, p=probs))
            if probs is not None
            else int(rng.integers(len(reader_titles)))
        )
        if doc in seen:
            continue
        seen.add(doc)
        title = reader_titles[doc]
        if reader.analyzer.terms(title):
            out.append((doc, title))
    return out


def candidates(reader, ranker, queries):
    """BM25 scores of every matching doc, computed once per query and reused for every weight."""
    out = []
    for doc, title in queries:
        weights = {}
        for group in parse(title, reader.analyzer):
            for clause in group:
                for t in clause.terms:
                    weights[t] = weights.get(t, 0) + 1
        acc = ranker.scores(weights)
        hits = np.flatnonzero(acc > 0)
        out.append((doc, hits, acc[hits]))
    return out


def score(cands, prior, w):
    s1 = mrr = 0.0
    for doc, hits, bm25 in cands:
        final = bm25 + w * prior[hits] if prior is not None else bm25
        top = hits[np.lexsort((hits, -final))[:10]]
        pos = np.flatnonzero(top == doc)
        if pos.size:
            mrr += 1 / (pos[0] + 1)
            s1 += pos[0] == 0
    n = len(cands)
    return {"success@1": round(s1 / n, 4), "MRR@10": round(mrr / n, 4)}


def first_hit(cands, prior, w) -> np.ndarray:
    """Per query: was the right article ranked first? (for paired significance tests)"""
    out = []
    for doc, hits, bm25 in cands:
        final = bm25 + w * prior[hits] if prior is not None else bm25
        out.append(bool(hits.size) and hits[np.lexsort((hits, -final))[0]] == doc)
    return np.array(out)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired outcomes: does b fix more queries than it breaks?"""
    fixed, broke = int((~a & b).sum()), int((a & ~b).sum())
    n, k = fixed + broke, min(fixed, broke)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"fixed": fixed, "broke": broke, "p_value": float(f"{p:.3g}")}


def bootstrap_gain(a: np.ndarray, b: np.ndarray, seed: int = 0, rounds: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    diffs = b.astype(int) - a.astype(int)
    boot = [diffs[rng.integers(0, diffs.size, diffs.size)].mean() for _ in range(rounds)]
    return [round(100 * float(np.percentile(boot, q)), 1) for q in (2.5, 97.5)]


reader_titles: list[str] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index-en")
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=1000)
    args = ap.parse_args()
    index = Path(args.index)

    reader = IndexReader(index)
    reader_titles.extend(t for (t,) in reader.db.execute("SELECT title FROM docs ORDER BY id"))
    ranker = BM25(reader, k1=1.2, b=0.75)
    views = np.load(index / "links.npz")["popularity"]
    probs = views / views.sum()
    rng = np.random.default_rng(42)

    train = sample(rng, args.train, probs, [], reader)
    test_traffic = sample(rng, args.test, probs, [d for d, _ in train], reader)
    test_uniform = sample(rng, args.test, None, [d for d, _ in train], reader)
    sets = {
        name: candidates(reader, ranker, qs)
        for name, qs in [("train", train), ("test_traffic", test_traffic), ("test_uniform", test_uniform)]
    }
    single_word = [
        c for c, (_, t) in zip(sets["test_traffic"], test_traffic, strict=True) if len(t.split()) == 1
    ]

    results = {
        "queries": {k: len(v) for k, v in sets.items()} | {"test_traffic_single_word": len(single_word)}
    }
    firsts: dict[str, np.ndarray] = {}
    for kind in ("none", "indegree", "pagerank"):
        prior = load_prior(index, kind)
        if prior is None:
            best_w, curve = 0.0, {}
        else:
            curve = {w: score(sets["train"], prior, w)["success@1"] for w in WEIGHTS}
            best_w = max(WEIGHTS, key=lambda w: (curve[w], -w))  # ties -> smaller weight
        results[kind] = {
            "tuned_weight": best_w,
            "train_curve_success@1": curve,
            "test_traffic": score(sets["test_traffic"], prior, best_w),
            "test_uniform": score(sets["test_uniform"], prior, best_w),
            "test_traffic_single_word": score(single_word, prior, best_w) if single_word else {},
        }
        print(kind, json.dumps(results[kind]), flush=True)
        firsts[kind] = first_hit(sets["test_traffic"], prior, best_w)
    results["significance_test_traffic"] = {
        "none_vs_indegree": mcnemar(firsts["none"], firsts["indegree"]),
        "none_vs_pagerank": mcnemar(firsts["none"], firsts["pagerank"]),
        "indegree_vs_pagerank": mcnemar(firsts["indegree"], firsts["pagerank"]),
        "pagerank_gain_pts_95ci": bootstrap_gain(firsts["none"], firsts["pagerank"]),
    }
    print(json.dumps(results["significance_test_traffic"]), flush=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "day04_prior.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
