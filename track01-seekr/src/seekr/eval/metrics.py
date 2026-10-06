"""Ranking metrics with trec_eval's conventions, so numbers compare to published baselines.

* Runs are re-sorted the way trec_eval does: score descending, ties broken by document id
  descending (string order). A run's own order is not trusted.
* nDCG uses gain = relevance grade and discount log2(rank + 1); the ideal ranking comes from
  every judged-relevant document for the query.
* Recall@k counts documents with grade > 0.
* `-c` semantics: every query in the qrels counts, including ones with no results (score 0).
"""

from __future__ import annotations

import math

Run = dict[str, dict[str, float]]  # query id -> doc id -> score
Qrels = dict[str, dict[str, int]]  # query id -> doc id -> relevance grade


def trec_order(docs: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def evaluate(run: Run, qrels: Qrels, ks_ndcg=(10,), ks_recall=(100, 1000)) -> dict[str, float]:
    totals: dict[str, float] = {f"nDCG@{k}": 0.0 for k in ks_ndcg} | {f"R@{k}": 0.0 for k in ks_recall}
    queries = [q for q, rels in qrels.items() if any(g > 0 for g in rels.values())]
    for q in queries:
        ranked = trec_order(run.get(q, {}))
        for k in ks_ndcg:
            totals[f"nDCG@{k}"] += ndcg_at_k(ranked, qrels[q], k)
        for k in ks_recall:
            totals[f"R@{k}"] += recall_at_k(ranked, qrels[q], k)
    return {m: v / max(1, len(queries)) for m, v in totals.items()} | {"queries": len(queries)}
