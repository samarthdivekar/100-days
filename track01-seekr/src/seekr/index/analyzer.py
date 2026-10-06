"""Analyzers: text -> (position, term) pairs, used identically at index and query time.

* ``plain``: Day 2's tokenizer. NFKC + casefold letter/digit runs, every token kept.
* ``english``: the pipeline of Lucene's EnglishAnalyzer, which Anserini's published BM25
  baselines use: strip possessive 's, drop Lucene's 33 English stopwords, then Porter-stem.

Positions are the token's place in the *original* stream, so removing a stopword leaves a gap
instead of shifting later words: the phrase "united states of america" still requires "america"
three positions after "united".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from seekr.index.tokenize import tokenize

# org.apache.lucene.analysis.en.EnglishAnalyzer.ENGLISH_STOP_WORDS_SET
LUCENE_STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such that the their then there
    these they this to was will with""".split()
)
_POSSESSIVE = re.compile(r"(\w)['’]s\b", re.UNICODE)

ANALYZERS = ("plain", "english")


@lru_cache(maxsize=1)
def _porter():
    import snowballstemmer

    return snowballstemmer.stemmer("porter")


@dataclass(frozen=True)
class Analyzer:
    name: str = "plain"

    def __post_init__(self):
        if self.name not in ANALYZERS:
            raise ValueError(f"unknown analyzer {self.name!r}; expected one of {ANALYZERS}")

    def analyze(self, text: str) -> list[tuple[int, str]]:
        if self.name == "plain":
            return list(enumerate(tokenize(text)))
        tokens = tokenize(_POSSESSIVE.sub(r"\1", text))
        kept = [(i, t) for i, t in enumerate(tokens) if t not in LUCENE_STOPWORDS]
        stems = _stem_all(tuple(t for _, t in kept))
        return [(i, s) for (i, _), s in zip(kept, stems, strict=True)]

    def terms(self, text: str) -> list[str]:
        return [t for _, t in self.analyze(text)]


_stem_cache: dict[str, str] = {}


def _stem_all(words: tuple[str, ...]) -> list[str]:
    """Porter-stem with a word cache: vocabularies repeat, so most lookups skip the stemmer."""
    out = []
    missing = [w for w in set(words) if w not in _stem_cache]
    if missing:
        for w, s in zip(missing, _porter().stemWords(missing), strict=True):
            _stem_cache[w] = s
    for w in words:
        out.append(_stem_cache[w])
    return out
