"""Boolean + phrase queries over the index.

    raptor hedges              AND of terms
    "new york" OR boston       phrase, OR binds tighter than the implicit AND
    enron -stock               NOT

AND intersects the shortest list first and searches the longer lists with binary search
(`searchsorted`), which costs O(m log n) instead of O(m + n) when lists differ a lot in length.
A phrase is one vectorized intersection over occurrence keys (doc << 32 | position): shift each
term's keys back by its offset in the phrase and intersect; whatever survives is a match.
Ranking (BM25) is Day 3; this module decides *which* documents match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from seekr.index.reader import IndexReader
from seekr.index.tokenize import tokenize

_EMPTY = np.zeros(0, dtype=np.int64)
_CLAUSE_RE = re.compile(r'(-?)"([^"]*)"|(\S+)')


@dataclass
class Clause:
    terms: list[str]  # one term, or several for a phrase
    negated: bool = False


def parse(query: str) -> list[list[Clause]]:
    """AND-list of OR-groups of clauses."""
    groups: list[list[Clause]] = []
    pending_or = False
    for m in _CLAUSE_RE.finditer(query):
        if m.group(3) == "OR":
            pending_or = bool(groups)
            continue
        if m.group(2) is not None:
            clause = Clause(tokenize(m.group(2)), negated=m.group(1) == "-")
        else:
            word = m.group(3)
            negated = word.startswith("-") and len(word) > 1
            clause = Clause(tokenize(word[1:] if negated else word), negated=negated)
        if not clause.terms:
            continue
        if pending_or and not clause.negated and not groups[-1][0].negated:
            groups[-1].append(clause)
        else:
            groups.append([clause])
        pending_or = False
    return groups


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    small, large = (a, b) if a.size <= b.size else (b, a)
    if small.size == 0:
        return _EMPTY
    idx = np.searchsorted(large, small)
    idx[idx == large.size] = large.size - 1
    return small[large[idx] == small]


class Searcher:
    def __init__(self, reader: IndexReader):
        self.r = reader

    def term_docs(self, term: str) -> np.ndarray:
        info = self.r.term(term)
        return self.r.docs(info) if info else _EMPTY

    def phrase_docs(self, terms: list[str]) -> np.ndarray:
        if len(terms) == 1:
            return self.term_docs(terms[0])
        infos = [self.r.term(t) for t in terms]
        if any(i is None for i in infos):
            return _EMPTY
        # Narrow to documents containing every term (rarest first) before decoding any positions.
        by_df = sorted(infos, key=lambda i: i.df)
        candidates = self.r.docs(by_df[0])
        for info in by_df[1:]:
            candidates = intersect(candidates, self.r.docs(info))
            if candidates.size == 0:
                return _EMPTY
        # Boolean table over doc ids: one O(n) gather instead of a sort-based np.isin.
        wanted = np.zeros(self.r.n_docs, dtype=bool)
        wanted[candidates] = True
        keys = None
        for offset, info in sorted(enumerate(infos), key=lambda x: x[1].cf):
            k = self.r.position_keys(info, candidates)
            k = k[wanted[k >> 32]] - offset
            keys = k if keys is None else np.intersect1d(keys, k, assume_unique=True)
            if keys.size == 0:
                return _EMPTY
        return np.unique(keys >> 32)

    def clause_docs(self, clause: Clause) -> np.ndarray:
        return self.phrase_docs(clause.terms)

    def search(self, query: str) -> np.ndarray:
        """Sorted ids of every matching document."""
        positive: list[np.ndarray] = []
        negative: list[np.ndarray] = []
        for group in parse(query):
            if group[0].negated:
                negative.append(self.clause_docs(group[0]))
                continue
            docs = self.clause_docs(group[0])
            for clause in group[1:]:
                docs = np.union1d(docs, self.clause_docs(clause))
            positive.append(docs)
        if positive:
            positive.sort(key=len)
            result = positive[0]
            for docs in positive[1:]:
                result = intersect(result, docs)
                if result.size == 0:
                    break
        elif negative:
            result = np.arange(self.r.n_docs, dtype=np.int64)
        else:
            return _EMPTY
        for docs in negative:
            result = np.setdiff1d(result, docs, assume_unique=True)
        return result
