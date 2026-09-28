"""Duplicate detection: exact (content hash) and near-duplicate (SimHash proposes, MinHash verifies).

A large share of the web is near-duplicate: the same article with a different timestamp, ad
slot or session id. Exact hashing misses those. SimHash (Charikar 2002) maps similar documents
to 64-bit fingerprints that differ in only a few bits; Google used it to dedup its crawl
(Manku, Jain & Das Sarma, WWW 2007) with a Hamming-distance threshold of 3.

What we measured (details in docs/day01-crawler.md):

1. k=3 does not transfer: on 300-word pages with one edited word plus a new timestamp it caught
   only 55% of near-duplicates. k=8 caught 98%.
2. On a real templated site (books.toscrape.com), k=8 over the full text flagged 27/200 pages,
   mostly category pages listing different books. Fingerprinting main content only (no nav,
   sidebar, footer; see parse.py) cut that to 19, still mostly wrong: every listing repeats
   "In stock / Add to basket / price" 20 times and those repeats outvote the book titles.

So the pipeline is now:

* Features are the SET of word 3-shingles (a repeated block counts once, not 20 times).
* SimHash over that set finds candidates cheaply (k-bit neighbours via the block index below).
* A candidate is only accepted if a 64-value MinHash signature estimates the shingle-set
  Jaccard similarity at >= 0.8. SimHash gives recall at low cost; the verifier gives precision.

Finding every stored fingerprint within distance k uses the paper's trick: split the 64 bits into
k+1 blocks. If two fingerprints differ in at most k bits, at least one block is identical
(pigeonhole), so we only compare against fingerprints sharing a block.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def content_hash(text: str) -> str:
    normalized = " ".join(_WORD_RE.findall(text.lower()))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1 << 20)
def _word_hash(word: str) -> int:
    """Stable 64-bit hash of one word (cached: vocabularies repeat, pages don't)."""
    return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")


_U64 = np.uint64


def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    return (x << _U64(r)) | (x >> _U64(64 - r))


def _mix(z: np.ndarray) -> np.ndarray:
    # splitmix64 finalizer: spreads every input bit over all 64 output bits (uint64 math wraps).
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return z ^ (z >> _U64(31))


_MINHASH_SEEDS = np.random.default_rng(20070508).integers(0, 2**63, size=64, dtype=np.uint64)


def shingles(text: str, shingle: int = 3) -> np.ndarray:
    """Sorted unique 64-bit hashes of the word `shingle`-grams in `text` (a set, as an array)."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return np.zeros(0, dtype=np.uint64)
    w = np.fromiter((_word_hash(x) for x in words), dtype=np.uint64, count=len(words))
    if len(w) >= shingle:
        n = len(w) - shingle + 1
        h = w[:n].copy()
        for j in range(1, shingle):
            h ^= _rotl(w[j : j + n], (21 * j) % 64)
    else:
        h = w
    return np.unique(_mix(h))


def simhash_of(features: np.ndarray) -> int:
    """64-bit SimHash of a feature set: each bit is the majority vote of that bit across features."""
    if features.size == 0:
        return 0
    bits = np.unpackbits(features.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")  # (n, 64)
    majority = bits.sum(axis=0, dtype=np.int64) * 2 > len(features)
    return int.from_bytes(np.packbits(majority, bitorder="little").tobytes(), "little")


def minhash_of(features: np.ndarray) -> np.ndarray:
    """64 MinHash values (uint32). P(two signatures agree at a position) = Jaccard of the sets."""
    if features.size == 0:
        return np.zeros(len(_MINHASH_SEEDS), dtype=np.uint32)
    table = _mix(features[None, :] ^ _MINHASH_SEEDS[:, None])  # (64, n)
    return (table.min(axis=1) & _U64(0xFFFFFFFF)).astype(np.uint32)


def jaccard_estimate(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


def fingerprint(text: str) -> tuple[int, np.ndarray]:
    """(SimHash, MinHash signature) of a text, both over its set of word 3-shingles."""
    features = shingles(text)
    return simhash_of(features), minhash_of(features)


def simhash(text: str) -> int:
    return simhash_of(shingles(text))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class NearDupIndex:
    """SimHash candidates within `k` bits (block index), accepted only if MinHash Jaccard >= `min_jaccard`."""

    def __init__(self, k: int = 8, min_jaccard: float = 0.8):
        self.k = k
        self.min_jaccard = min_jaccard
        self.blocks = k + 1
        self.width = 64 // self.blocks
        self._tables: list[dict[int, list[tuple[int, str]]]] = [defaultdict(list) for _ in range(self.blocks)]
        self._sigs: dict[str, np.ndarray] = {}
        self.rejected = 0  # SimHash candidates the MinHash check turned down

    def _block(self, fp: int, i: int) -> int:
        shift = i * self.width
        width = 64 - shift if i == self.blocks - 1 else self.width
        return (fp >> shift) & ((1 << width) - 1)

    def candidates(self, fp: int) -> list[str]:
        out: dict[str, None] = {}
        for i, table in enumerate(self._tables):
            for other, key in table.get(self._block(fp, i), ()):
                if hamming(fp, other) <= self.k:
                    out[key] = None
        return list(out)

    def find(self, fp: int, sig: np.ndarray | None = None) -> str | None:
        for key in self.candidates(fp):
            if sig is None or jaccard_estimate(sig, self._sigs[key]) >= self.min_jaccard:
                return key
            self.rejected += 1
        return None

    def add(self, fp: int, key: str, sig: np.ndarray | None = None) -> None:
        for i, table in enumerate(self._tables):
            table[self._block(fp, i)].append((fp, key))
        if sig is not None:
            self._sigs[key] = sig
