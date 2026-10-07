"""BM25 ranking (Robertson & Zaragoza), scored the way Lucene/Anserini do.

    score(d, q) = Σ_t  w_t · idf(t) · tf / (tf + k1 · (1 − b + b · |d| / avgdl))
    idf(t)      = ln(1 + (N − df + 0.5) / (df + 0.5))

w_t is how many times t occurs in the query (Anserini's bag-of-words query boosts each unique
term by its count). Lucene ≥ 8 dropped the constant (k1 + 1) factor; it doesn't change ranking.

Lucene does not keep exact document lengths: each is squeezed into ONE byte
(`SmallFloat.intToByte4`, exact below 24, then 4 significant bits). `lucene_norms=True` applies the
same quantization so scores line up with Anserini's published baselines; docs/day03-bm25.md
measures what difference it makes.

Scoring is term-at-a-time into a dense float32 accumulator (one vectorized update per query term),
then a partial sort (`argpartition`) for the top k.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from seekr.index.reader import IndexReader

# Lucene SmallFloat.intToByte4 / byte4ToInt
_NUM_FREE_VALUES = 24


def _long_to_int4(i: int) -> int:
    bits = i.bit_length()
    if bits < 4:
        return i
    shift = bits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i: int) -> int:
    bits = i & 0x07
    shift = (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


def lucene_length(length: int) -> int:
    """The document length Lucene's BM25 actually sees after its 1-byte norm encoding."""
    if length < _NUM_FREE_VALUES:
        return length
    return _NUM_FREE_VALUES + _int4_to_long(_long_to_int4(length - _NUM_FREE_VALUES))


class BM25:
    def __init__(self, reader: IndexReader, k1: float = 0.9, b: float = 0.4, lucene_norms: bool = True):
        self.r = reader
        self.k1, self.b = k1, b
        lengths = reader.doc_lengths()
        self.n_docs = reader.n_docs
        self.avgdl = float(lengths.sum()) / max(1, self.n_docs)  # Lucene: sumTotalTermFreq / docCount
        if lucene_norms:
            table = np.array(
                [lucene_length(x) for x in range(int(lengths.max(initial=0)) + 1)], dtype=np.float64
            )
            lengths = table[lengths]
        self.norm = (k1 * (1 - b + b * lengths / self.avgdl)).astype(np.float32)

    def idf(self, df: int) -> float:
        return float(np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)))

    def query_weights(self, text: str) -> Counter:
        return Counter(self.r.analyzer.terms(text))

    def scores(self, weights: Counter) -> np.ndarray:
        acc = np.zeros(self.n_docs, dtype=np.float32)
        for term, w in weights.items():
            info = self.r.term(term)
            if info is None:
                continue
            docs = self.r.docs(info)
            tfs = self.r.tfs(info).astype(np.float32)
            # docs are unique within a postings list, so a fancy-index += is safe
            acc[docs] += np.float32(w * self.idf(info.df)) * tfs / (tfs + self.norm[docs])
        return acc

    def search(
        self, text: str, k: int = 10, candidates: np.ndarray | None = None, exclude: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Top-k (doc id, score), best first; ties broken by lower doc id.

        `candidates` restricts ranking to those docs (e.g. the result of a boolean/phrase filter).
        """
        acc = self.scores(self.query_weights(text))
        if candidates is not None:
            mask = np.zeros(self.n_docs, dtype=bool)
            mask[candidates] = True
            acc[~mask] = 0
        for doc in exclude or ():
            acc[doc] = 0
        hits = np.flatnonzero(acc > 0)
        if hits.size > k:
            top = np.argpartition(-acc[hits], k - 1)[:k]
            hits = hits[top]
        order = np.lexsort((hits, -acc[hits]))
        return [(int(d), float(acc[d])) for d in hits[order]]


def ranked_search(
    reader: IndexReader,
    query: str,
    k: int = 10,
    ranker: BM25 | None = None,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> list[tuple[int, float]]:
    """Seekr's search: BM25 over the query's positive terms.

    Plain words rank the whole corpus (any word may match). Phrases, `-exclusions` and `OR` keep their
    boolean meaning: they decide which documents qualify, and BM25 orders those. With a `prior`
    (e.g. log PageRank), matching documents score bm25 + prior_weight * prior.
    """
    from seekr.index.query import Searcher, parse

    ranker = ranker or BM25(reader, k1=1.2, b=0.75)  # product defaults; see docs/day03-bm25.md
    groups = parse(query, reader.analyzer)
    weights: Counter = Counter()
    for group in groups:
        if not group[0].negated:
            for clause in group:
                weights.update(clause.terms)
    if not weights:
        return []
    structured = any(len(c.terms) > 1 or c.negated for g in groups for c in g) or any(
        len(g) > 1 for g in groups
    )
    candidates = Searcher(reader).search(query) if structured else None
    acc = ranker.scores(weights)
    if candidates is not None:
        mask = np.zeros(ranker.n_docs, dtype=bool)
        mask[candidates] = True
        acc[~mask] = 0
    hits = np.flatnonzero(acc > 0)
    final = acc[hits] + prior_weight * prior[hits] if prior is not None and prior_weight else acc[hits]
    if hits.size > k:
        top = np.argpartition(-final, k - 1)[:k]
        hits, final = hits[top], final[top]
    order = np.lexsort((hits, -final))
    return [(int(d), float(s)) for d, s in zip(hits[order], final[order], strict=True)]


def load_prior(index_dir, kind: str) -> np.ndarray | None:
    """Static, query-independent document scores to add to BM25 (Day 4).

    pagerank: log(PageRank · N), so an average page scores 0 and each 10x in PageRank adds ln 10.
    indegree: log(1 + in-links), the simple baseline PageRank has to beat.
    """
    from pathlib import Path

    index_dir = Path(index_dir)
    if kind in ("none", None):
        return None
    if kind == "pagerank":
        pr = np.load(index_dir / "pagerank.npy")
        return np.log(pr * pr.size).astype(np.float32)
    if kind == "indegree":
        links = np.load(index_dir / "links.npz")
        n = len(links["incoming_links"])
        return np.log1p(np.bincount(links["dst"], minlength=n)).astype(np.float32)
    raise ValueError(f"unknown prior {kind!r}; expected none, pagerank or indegree")
