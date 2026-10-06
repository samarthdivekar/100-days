"""Day 3: analyzer, BM25 (vs a brute-force scorer), trec_eval-style metrics, BEIR loading."""

import json
import math
import random
from collections import Counter

import numpy as np
import pytest

from seekr.eval import beir
from seekr.eval.metrics import evaluate, ndcg_at_k, recall_at_k, trec_order
from seekr.index.analyzer import Analyzer
from seekr.index.builder import Document, IndexBuilder
from seekr.index.query import Searcher
from seekr.index.rank import BM25, lucene_length
from seekr.index.reader import IndexReader

ENGLISH = Analyzer("english")


# ------------------------------------------------------------------ analyzer


def test_english_analyzer_stops_stems_and_keeps_position_gaps():
    pairs = ENGLISH.analyze("The Studies of Enron's connections")
    assert pairs == [(1, "studi"), (3, "enron"), (4, "connect")]  # "the"/"of" gone, gaps kept


def test_plain_analyzer_is_day2_tokenizer():
    assert Analyzer("plain").terms("The cats' toys") == ["the", "cats", "toys"]


def test_unknown_analyzer_rejected():
    with pytest.raises(ValueError):
        Analyzer("klingon")


# ------------------------------------------------------------------ Lucene norm quantization


def test_lucene_length_quantization():
    assert [lucene_length(n) for n in range(24)] == list(range(24))  # exact below 24
    assert lucene_length(100) == 96
    prev = 0
    for n in range(1, 20000):
        q = lucene_length(n)
        assert prev <= q <= n and (n - q) / n <= 1 / 8  # monotone, never overshoots, <=12.5% off
        prev = q


# ------------------------------------------------------------------ BM25 vs brute force

VOCAB = "enron raptor hedge ljm stock swap gas power trade risk energy deal board loss audit".split()


@pytest.fixture(scope="module")
def corpus_index(tmp_path_factory):
    rng = random.Random(3)
    docs = [
        Document(f"d{i}", "", " ".join(rng.choice(VOCAB) for _ in range(rng.randint(3, 120))))
        for i in range(300)
    ]
    out = tmp_path_factory.mktemp("rank") / "idx"
    b = IndexBuilder(out, block_tokens=2000)
    b.add_all(docs)
    b.finish()
    reader = IndexReader(out)
    yield docs, reader
    reader.close()


def _brute_bm25(docs, query_terms, k1, b, lucene_norms):
    toks = [Analyzer("plain").terms(d.text) for d in docs]
    lengths = [len(t) for t in toks]
    avgdl = sum(lengths) / len(lengths)
    n = len(docs)
    scores = []
    for t, dl in zip(toks, lengths, strict=True):
        dl = lucene_length(dl) if lucene_norms else dl
        counts = Counter(t)
        s = 0.0
        for term, w in Counter(query_terms).items():
            df = sum(1 for x in toks if term in x)
            if counts[term] and df:
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = counts[term]
                s += w * idf * tf / (tf + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return np.array(scores)


@pytest.mark.parametrize("lucene_norms", [True, False])
@pytest.mark.parametrize("k1,b", [(0.9, 0.4), (1.2, 0.75)])
def test_bm25_matches_brute_force(corpus_index, lucene_norms, k1, b):
    docs, reader = corpus_index
    ranker = BM25(reader, k1=k1, b=b, lucene_norms=lucene_norms)
    rng = random.Random(9)
    for _ in range(40):
        q = rng.sample(VOCAB, rng.randint(1, 4)) + ([rng.choice(VOCAB)] if rng.random() < 0.3 else [])
        expected = _brute_bm25(docs, q, k1, b, lucene_norms)
        got = ranker.scores(Counter(q))
        assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)
        top = ranker.search(" ".join(q), k=10)
        want = sorted(expected[expected > 0], reverse=True)[:10]
        # Lucene's length quantization makes exact score ties common, and float32 vs float64 may
        # order a tie differently, so check that rank i holds a document with the i-th best score.
        assert np.allclose([expected[d] for d, _ in top], want, rtol=1e-5)


def test_candidates_and_exclude(corpus_index):
    docs, reader = corpus_index
    ranker = BM25(reader)
    allowed = Searcher(reader).search('"enron raptor"')
    ranked = ranker.search("enron raptor", k=50, candidates=allowed)
    assert ranked and {d for d, _ in ranked} <= set(allowed.tolist())
    best = ranker.search("enron raptor", k=1)[0][0]
    assert best not in {d for d, _ in ranker.search("enron raptor", k=5, exclude={best})}


def test_english_phrase_respects_stopword_gaps(tmp_path):
    docs = [
        Document("a", "", "the united states of america"),
        Document("b", "", "united states america"),
    ]
    b = IndexBuilder(tmp_path / "en", analyzer="english")
    b.add_all(docs)
    b.finish()
    r = IndexReader(tmp_path / "en")
    s = Searcher(r)
    assert s.search('"states of america"').tolist() == [0]  # the gap where "of" was must be there
    assert s.search('"united states"').tolist() == [0, 1]
    assert r.analyzer.name == "english"
    r.close()


# ------------------------------------------------------------------ metrics


def test_trec_order_breaks_ties_by_docid_descending():
    assert trec_order({"a": 1.0, "c": 1.0, "b": 2.0}) == ["b", "c", "a"]


def test_ndcg_and_recall_by_hand():
    rels = {"x": 2, "y": 1, "z": 0}
    assert ndcg_at_k(["x", "y"], rels, 10) == pytest.approx(1.0)
    dcg = 0 + 2 / math.log2(3)
    idcg = 2 + 1 / math.log2(3)
    assert ndcg_at_k(["n", "x"], rels, 10) == pytest.approx(dcg / idcg)
    assert recall_at_k(["y", "n"], rels, 1) == 0.5


def test_evaluate_counts_queries_without_results():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    m = evaluate({"q1": {"a": 3.0}}, qrels)
    assert m["queries"] == 2 and m["nDCG@10"] == pytest.approx(0.5)


# ------------------------------------------------------------------ BEIR runner


def test_beir_runner_removes_query_document(tmp_path):
    d = tmp_path / "toy"
    (d / "qrels").mkdir(parents=True)
    corpus = [
        {
            "_id": "q1",
            "title": "",
            "text": "nuclear power plants are safe",
        },  # the query itself, as in ArguAna
        {"_id": "d1", "title": "Reactor safety", "text": "nuclear power plant safety record"},
        {"_id": "d2", "title": "", "text": "solar panels on roofs"},
    ]
    (d / "corpus.jsonl").write_text("\n".join(json.dumps(x) for x in corpus))
    (d / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "nuclear power plants are safe"}))
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    beir.build(d, tmp_path / "idx", "english")
    run, _ = beir.run_bm25(d, tmp_path / "idx")
    assert "q1" not in run["q1"] and next(iter(trec_order(run["q1"]))) == "d1"
    assert beir.evaluate_dataset(d, run)["nDCG@10"] == pytest.approx(1.0)


def test_ranked_search_filters_then_ranks(corpus_index):
    from seekr.index.rank import ranked_search

    docs, reader = corpus_index
    free = ranked_search(reader, "enron raptor", k=300)
    phrase = ranked_search(reader, '"enron raptor"', k=300)
    negated = ranked_search(reader, "enron -raptor", k=300)
    allowed = set(Searcher(reader).search('"enron raptor"').tolist())
    assert {d for d, _ in phrase} == allowed  # the phrase decides membership
    assert len(free) > len(phrase)  # plain words: any word may match
    assert all("raptor" not in docs[d].text.split() for d, _ in negated)
    assert [s for _, s in free] == sorted((s for _, s in free), reverse=True)
