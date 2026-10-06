"""BEIR datasets (Thakur et al., NeurIPS 2021): corpus, queries, relevance judgments, and a runner.

Indexing mirrors Anserini's "flat" BEIR setup: title and text concatenated into one field.
Like Anserini's `-removeQuery`, a query never retrieves the document with its own id (matters for
ArguAna, where queries are themselves corpus arguments).
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator
from pathlib import Path

from seekr.eval.metrics import Qrels, Run, evaluate
from seekr.index.builder import Document, IndexBuilder
from seekr.index.rank import BM25
from seekr.index.reader import IndexReader

# Anserini BM25 "flat" baselines, nDCG@10 / R@100 / R@1000 (k1=0.9, b=0.4, Lucene EnglishAnalyzer).
# Source: castorini/anserini docs/reproduce/from-document-collection/beir-v1.0.0-<name>.flat.md
ANSERINI_BM25_FLAT = {
    "scifact": (0.6789, 0.9253, 0.9767),
    "nfcorpus": (0.3218, 0.2457, 0.3704),
    "arguana": (0.3970, 0.9324, 0.9872),
    "fiqa": (0.2361, 0.5395, 0.7393),
    "trec-covid": (0.5947, 0.1091, 0.3955),
}


def corpus(path: Path) -> Iterator[Document]:
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield Document(url=d["_id"], title=d.get("title") or "", text=d.get("text") or "")


def queries(path: Path) -> dict[str, str]:
    out = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            out[q["_id"]] = q["text"]
    return out


def qrels(path: Path, split: str = "test") -> Qrels:
    out: Qrels = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return out


def build(dataset_dir: Path, index_dir: Path, analyzer: str, workers: int = 0) -> dict:
    builder = IndexBuilder(index_dir, analyzer=analyzer, workers=workers, store_text=False)
    builder.add_all(corpus(dataset_dir))
    return builder.finish()


def run_bm25(
    dataset_dir: Path,
    index_dir: Path,
    k1: float = 0.9,
    b: float = 0.4,
    lucene_norms: bool = True,
    hits: int = 1000,
) -> tuple[Run, dict]:
    reader = IndexReader(index_dir)
    ids = [u for (u,) in reader.db.execute("SELECT url FROM docs ORDER BY id")]
    by_id = {doc_id: i for i, doc_id in enumerate(ids)}
    ranker = BM25(reader, k1=k1, b=b, lucene_norms=lucene_norms)
    texts = queries(dataset_dir)
    judged = qrels(dataset_dir)
    run: Run = {}
    latencies = []
    for qid in judged:
        exclude = {by_id[qid]} if qid in by_id else set()
        t0 = time.perf_counter()
        results = ranker.search(texts[qid], k=hits, exclude=exclude)
        latencies.append((time.perf_counter() - t0) * 1000)
        run[qid] = {ids[d]: s for d, s in results}
    reader.close()
    latencies.sort()
    timing = {"p50_ms": latencies[len(latencies) // 2], "p99_ms": latencies[int(0.99 * (len(latencies) - 1))]}
    return run, timing


def evaluate_dataset(dataset_dir: Path, run: Run) -> dict:
    return evaluate(run, qrels(dataset_dir))
