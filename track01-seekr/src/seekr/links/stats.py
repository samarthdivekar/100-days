"""Rank statistics for comparing importance signals."""

from __future__ import annotations

import numpy as np


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (like scipy.stats.rankdata)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    # group boundaries of equal values
    starts = np.flatnonzero(np.r_[True, sorted_x[1:] != sorted_x[:-1]])
    ends = np.r_[starts[1:], x.size]
    avg = (starts + ends + 1) / 2.0  # mean of 1-based positions starts+1 .. ends
    ranks = np.empty(x.size)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Share of the top-k items by `a` that are also in the top-k by `b`."""
    ta = set(np.argsort(-a, kind="stable")[:k].tolist())
    tb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(ta & tb) / k
