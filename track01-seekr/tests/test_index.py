"""Inverted index: codec round-trips, and every query type checked against a brute-force scan."""

import random

import numpy as np
import pytest

from seekr.index import vbyte
from seekr.index.builder import Document, IndexBuilder
from seekr.index.query import Searcher, parse
from seekr.index.reader import IndexReader
from seekr.index.tokenize import tokenize

VOCAB = "enron raptor hedge ljm stock swap gas power trade risk new york city energy deal board".split()


@pytest.mark.parametrize(
    "values",
    [[0], [127], [128], [16383], [16384], [2**32 - 1], [2**32], [2**63 - 1], [2**64 - 1], list(range(1000))],
)
def test_vbyte_roundtrip_edges(values):
    arr = np.array(values, dtype=np.uint64)
    assert np.array_equal(vbyte.decode(vbyte.encode(arr)), arr)


def test_vbyte_sizes_and_gaps():
    assert len(vbyte.encode(np.array([127], dtype=np.uint64))) == 1
    assert len(vbyte.encode(np.array([128], dtype=np.uint64))) == 2
    rng = np.random.default_rng(0)
    ids = np.unique(rng.integers(0, 10_000_000, 50_000)).astype(np.uint64)
    assert np.array_equal(vbyte.decode_gaps(vbyte.encode_gaps(ids)), ids)
    assert len(vbyte.encode_gaps(ids)) < 2.5 * ids.size  # gaps average ~200 -> ~2 bytes each


def test_tokenize():
    assert tokenize("Ｅｎｒｏｎ's Raptor-II, ÉNERGY_2001!") == [
        "enron",
        "s",
        "raptor",
        "ii",
        "énergy",
        "2001",
    ]


def test_parse_precedence():
    groups = parse('raptor "new york" OR boston -stock')
    assert [[c.terms for c in g] for g in groups] == [
        [["raptor"]],
        [["new", "york"], ["boston"]],
        [["stock"]],
    ]
    assert groups[-1][0].negated


def _corpus(n_docs=400, seed=1):
    rng = random.Random(seed)
    docs = []
    for i in range(n_docs):
        words = [rng.choice(VOCAB) for _ in range(rng.randint(5, 60))]
        docs.append(Document(url=f"http://x/{i}", title=f"Doc {i}", text=" ".join(words)))
    return docs


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    docs = _corpus()
    out = tmp_path_factory.mktemp("idx")
    builder = IndexBuilder(out / "multi", block_tokens=500)  # forces many blocks and a real merge
    builder.add_all(docs)
    stats = builder.finish()
    reader = IndexReader(out / "multi")
    yield docs, reader, stats, out
    reader.close()


def _brute(docs, query):
    """Evaluate a query by scanning token lists: the oracle the index must agree with."""
    toks = [tokenize(f"{d.title}\n{d.text}") for d in docs]

    def has(t, terms):
        n = len(terms)
        return any(t[i : i + n] == terms for i in range(len(t) - n + 1))

    result = None
    negatives = []
    for group in parse(query):
        if group[0].negated:
            negatives.append({i for i, t in enumerate(toks) if has(t, group[0].terms)})
            continue
        docs_g = {i for i, t in enumerate(toks) if any(has(t, c.terms) for c in group)}
        result = docs_g if result is None else result & docs_g
    if result is None:
        result = set(range(len(docs))) if negatives else set()
    for neg in negatives:
        result -= neg
    return sorted(result)


def test_queries_match_brute_force(built):
    docs, reader, stats, _ = built
    assert stats["blocks"] > 10
    searcher = Searcher(reader)
    rng = random.Random(7)
    queries = []
    for _ in range(300):
        kind = rng.random()
        a, b, c = rng.sample(VOCAB, 3)
        if kind < 0.25:
            queries.append(f"{a} {b}")
        elif kind < 0.5:
            queries.append(f'"{a} {b}"')
        elif kind < 0.65:
            queries.append(f'"{a} {b} {c}"')
        elif kind < 0.8:
            queries.append(f"{a} {b} OR {c}")
        elif kind < 0.9:
            queries.append(f"{a} -{b}")
        else:
            queries.append(f'-{a} "{b} {c}"')
    queries += ["nonexistentterm", '"enron nonexistentterm"', "-enron"]
    for q in queries:
        assert searcher.search(q).tolist() == _brute(docs, q), q


def test_multi_block_build_equals_single_block(built):
    docs, reader, _, out = built
    single = IndexBuilder(out / "single", block_tokens=10**9)
    single.add_all(docs)
    single.finish()
    assert (out / "single" / "postings.bin").read_bytes() == (out / "multi" / "postings.bin").read_bytes()
    one = IndexReader(out / "single")
    q = "SELECT * FROM terms ORDER BY term"
    assert one.db.execute(q).fetchall() == reader.db.execute(q).fetchall()
    one.close()


def test_positions_doc_lengths_and_doc_store(built):
    docs, reader, stats, _ = built
    toks = [tokenize(f"{d.title}\n{d.text}") for d in docs]
    info = reader.term("enron")
    keys = reader.position_keys(info)
    expected = sorted((i << 32) | p for i, t in enumerate(toks) for p, w in enumerate(t) if w == "enron")
    assert keys.tolist() == expected
    assert reader.doc_lengths().tolist() == [len(t) for t in toks]
    assert reader.doc(3)["text"] == docs[3].text
    assert stats["postings_bytes"] < stats["uncompressed_bytes"] / 3


def test_index_a_day1_crawl(tmp_path):
    import asyncio

    from seekr.crawl.crawler import CrawlConfig, Crawler
    from seekr.crawl.simweb import SimWeb
    from seekr.crawl.store import PageStore
    from seekr.index.corpus import from_crawl

    web = SimWeb(n_hosts=2, pages_per_host=15, latency=0.001, seed=8)
    cfg = CrawlConfig(seeds=[f"http://{h}/page/0" for h in web.hosts], max_pages=500, delay=0.0)
    store = PageStore(tmp_path / "crawl.db")
    asyncio.run(Crawler(cfg, store, web.transport()).run())
    store.close()
    builder = IndexBuilder(tmp_path / "idx")
    builder.add_all(from_crawl(tmp_path / "crawl.db"))
    stats = builder.finish()
    reader = IndexReader(tmp_path / "idx")
    kept = stats["docs"]
    crawled_ok = sum(1 for _ in from_crawl(tmp_path / "crawl.db"))
    assert kept == crawled_ok > 0  # duplicates were left out of the index
    word = web.pages["http://site0.test/page/0"].split()[0]
    assert Searcher(reader).search(word).size >= 1
    reader.close()


def test_parallel_build_is_byte_identical(built, tmp_path):
    docs, _, _, out = built
    par = IndexBuilder(tmp_path / "par", block_tokens=500, workers=2)
    par.add_all(docs)
    par.finish()
    assert (tmp_path / "par" / "postings.bin").read_bytes() == (out / "multi" / "postings.bin").read_bytes()


def test_skip_tables_give_identical_phrase_results(built, tmp_path):
    """Force skip tables on every term (tiny chunks) and re-run the brute-force comparison."""
    import shutil

    from seekr.index.skips import build_skips

    docs, _, _, out = built
    shutil.copytree(out / "multi", tmp_path / "sk")
    build_skips(tmp_path / "sk", min_df=2, every=4)
    reader = IndexReader(tmp_path / "sk")
    assert reader.skip_every == 4
    searcher = Searcher(reader)
    info = reader.term("enron")
    full = reader.position_keys(info)
    cand = reader.docs(info)[::7]
    part = reader.position_keys(info, cand)
    assert set(full[np.isin(full >> 32, cand)].tolist()) <= set(part.tolist()) <= set(full.tolist())
    rng = random.Random(11)
    for _ in range(150):
        a, b, c = rng.sample(VOCAB, 3)
        q = f'"{a} {b}"' if rng.random() < 0.5 else f'"{a} {b} {c}" {rng.choice(VOCAB)}'
        assert searcher.search(q).tolist() == _brute(docs, q), q
    reader.close()
