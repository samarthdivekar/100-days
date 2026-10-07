"""PageRank by power iteration over an edge list (Brin & Page, 1998).

    r' = d · Aᵀ (r / outdeg)  +  d · (Σ dangling r) / N  +  (1 − d) / N

* A random surfer follows a link with probability d (0.85) or jumps to a random page.
* Dangling pages (no out-links) would leak probability mass; their rank is spread evenly over
  all pages, as in networkx and the original formulation, so ranks always sum to 1.
* One iteration is two vectorized steps: gather r[src] / outdeg[src] per edge, then scatter-add
  into targets with `np.bincount(dst, weights=...)`. No sparse-matrix library needed.

Convergence: stop when the L1 change between iterations falls below N · tol (networkx's rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageRankResult:
    rank: np.ndarray
    iterations: int
    converged: bool
    l1_history: list[float] = field(default_factory=list)


def pagerank(
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
    damping: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> PageRankResult:
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    outdeg = np.bincount(src, minlength=n).astype(np.float64)
    dangling = outdeg == 0
    inv_out = np.zeros(n)
    inv_out[~dangling] = 1.0 / outdeg[~dangling]
    edge_weight = inv_out[src]  # constant across iterations

    rank = np.full(n, 1.0 / n)
    history: list[float] = []
    for it in range(1, max_iter + 1):
        spread = np.bincount(dst, weights=rank[src] * edge_weight, minlength=n)
        new = damping * (spread + rank[dangling].sum() / n) + (1.0 - damping) / n
        err = float(np.abs(new - rank).sum())
        history.append(err)
        rank = new
        if err < n * tol:
            return PageRankResult(rank, it, True, history)
    return PageRankResult(rank, max_iter, False, history)


def in_degree(dst: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(np.asarray(dst, dtype=np.int64), minlength=n)
