"""Day 4: PageRank vs networkx, rank statistics vs SciPy, link-graph building on a crafted dump."""

import gzip
import json
import random

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr

from seekr.links.graph import build_link_graph, normalize_title
from seekr.links.pagerank import in_degree, pagerank
from seekr.links.stats import average_ranks, spearman, top_k_overlap


def _random_graph(n, m, seed, dangling_share=0.2):
    rng = random.Random(seed)
    dangling = set(rng.sample(range(n), int(n * dangling_share)))
    edges = set()
    while len(edges) < m:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b and a not in dangling:
            edges.add((a, b))
    return sorted(edges)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("damping", [0.85, 0.5])
def test_pagerank_matches_networkx(seed, damping):
    n = 300
    edges = _random_graph(n, 1500, seed)
    src, dst = np.array(edges).T
    ours = pagerank(src, dst, n, damping=damping, tol=1e-12)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    ref = nx.pagerank(g, alpha=damping, tol=1e-12, max_iter=500)
    assert ours.converged
    assert np.allclose(ours.rank, [ref[i] for i in range(n)], atol=1e-9)
    assert ours.rank.sum() == pytest.approx(1.0)


def test_pagerank_basics():
    # A cycle is symmetric: everyone gets 1/n.
    r = pagerank(np.array([0, 1, 2]), np.array([1, 2, 0]), 3).rank
    assert np.allclose(r, 1 / 3)
    # A page everyone links to outranks the pages linking to it; isolated page 4 is dangling,
    # and its rank is redistributed instead of leaking, so ranks still sum to 1.
    src, dst = np.array([1, 2, 3, 0]), np.array([0, 0, 0, 1])
    res = pagerank(src, dst, 5)
    assert res.rank.argmax() == 0 and res.rank.sum() == pytest.approx(1.0)
    # A page that passes all its rank on to a single target lifts that target above itself.
    chain = pagerank(np.array([1, 2, 3, 0]), np.array([0, 0, 0, 4]), 5).rank
    assert chain[4] > chain[0]
    assert res.l1_history[-1] < res.l1_history[0]


def test_in_degree():
    assert in_degree(np.array([0, 1, 2]), 4).tolist() == [1, 1, 1, 0]
    assert in_degree(np.array([3, 3, 1]), 4).tolist() == [0, 1, 0, 2]


def test_rank_statistics_match_scipy():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 20, 500)  # many ties, like in-degree
    b = a + rng.normal(0, 5, 500)
    assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)
    assert average_ranks(np.array([10, 20, 20, 5])).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert top_k_overlap(np.array([5, 4, 3, 2]), np.array([4, 5, 1, 9]), 2) == 0.5


def test_normalize_title():
    assert normalize_title("united_states") == "United states"
    assert normalize_title(" New_York ") == "New York"


def _dump(path, docs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({"index": {"_id": str(i)}}) + "\n")
            f.write(json.dumps(d) + "\n")


def test_build_link_graph_resolves_redirects_and_drops_noise(tmp_path):
    docs = [
        {
            "title": "United States",
            "namespace": 0,
            "text": "a country",
            "incoming_links": 9,
            "popularity_score": 0.5,
            "redirect": [{"namespace": 0, "title": "USA"}],
            "outgoing_link": ["Canada", "United_States", "Canada"],
        },  # self-link + duplicate
        {
            "title": "Canada",
            "namespace": 0,
            "text": "another country",
            "incoming_links": 3,
            "popularity_score": 0.2,
            "outgoing_link": ["USA", "Atlantis"],
        },  # via redirect + red link
        {"title": "Talk page", "namespace": 1, "text": "not an article", "outgoing_link": ["Canada"]},
        {"title": "Empty", "namespace": 0, "text": "", "outgoing_link": ["Canada"]},  # not indexed
        {"title": "mexico", "namespace": 0, "text": "third", "outgoing_link": ["united_States", "canada"]},
    ]
    _dump(tmp_path / "d.json.gz", docs)
    g = build_link_graph(tmp_path / "d.json.gz")
    assert g.titles == ["United States", "Canada", "Mexico"]  # same ids the index assigns
    assert sorted(zip(g.src.tolist(), g.dst.tolist(), strict=True)) == [(0, 1), (1, 0), (2, 0), (2, 1)]
    assert g.stats["self_links"] == 1 and g.stats["unresolved_links"] == 1
    assert g.incoming_links.tolist() == [9, 3, 0]
    assert g.popularity.tolist() == [0.5, 0.2, 0.0]
    g.save(tmp_path / "links.npz")
    back = type(g).load(tmp_path / "links.npz")
    assert back.n == 3 and back.src.tolist() == g.src.tolist()
