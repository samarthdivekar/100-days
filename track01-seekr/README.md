# Seekr

A web search engine built from scratch over 10 days: crawler, compressed inverted index, BM25,
PageRank, spelling correction, autocomplete, HNSW vector search, learning to rank, and a sharded
serving layer. Track 1 of [100 Days, 100 Projects](../README.md).

| Day | Component | Status | Headline result |
|---|---|---|---|
| 1 | [Polite, deduplicating crawler](docs/day01-crawler.md) | ✅ | 55 pages/s at 82.5% of the politeness ceiling; 0 robots/politeness violations; duplicate precision 1.00, recall 0.986 |
| 2 | [Compressed positional inverted index](docs/day02-index.md) | ✅ | 278k Wikipedia articles in 8 min; 3.2× compression; term p50 1.5 ms, AND 6.9 ms, phrase 60 ms; matches brute force |
| 3 | [BM25 ranking, checked against Anserini](docs/day03-bm25.md) | ✅ | Within 0.0045 nDCG@10 of Anserini on 5 BEIR datasets; on Wikipedia the right article ranks first 88% of the time (unranked: 15%) |
| 4 | PageRank | ⏳ | |
| 5 | Spelling correction | ⏳ | |
| 6 | Autocomplete | ⏳ | |
| 7 | HNSW vector index | ⏳ | |
| 8 | Learning to rank | ⏳ | |
| 9 | Sharded search | ⏳ | |
| 10 | **Seekr**: everything + UI + load test | ⏳ | |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest                                    # 67 tests, offline
python bench/bench_crawler.py                       # synthetic-web benchmark
python -m seekr.cli crawl https://books.toscrape.com/ --max-pages 200 --delay 1.0 --db data/books.db
```

## Layout

```
src/seekr/crawl/
  urls.py      URL normalization
  robots.py    robots.txt, RFC 9309 (longest match, wildcards, unreachable-file rules)
  parse.py     HTML -> text, main content, links, meta robots, canonical
  dedup.py     content hash, SimHash (vectorized) + block index, MinHash verification
  crawler.py   per-host politeness scheduler, async workers, retries, optional process pool
  store.py     SQLite: pages, link graph, compressed HTML
  simweb.py    synthetic web with ground truth, for tests and benchmarks
src/seekr/index/
  tokenize.py  NFKC + casefold tokens with positions
  vbyte.py     VByte codec: Python for short lists, NumPy for long ones
  builder.py   parallel SPIMI: batches -> sorted compressed blocks -> k-way merge
  skips.py     position skip tables for frequent terms
  reader.py    memory-mapped postings, lexicon, doc table
  query.py     AND / OR / NOT / "phrase"
  analyzer.py  plain or Lucene-style English (stopwords + Porter), positions kept
  rank.py      BM25 (Lucene's formula, optional 1-byte length norms), ranked search
  corpus.py    Day 1 crawl DB and Wikimedia CirrusSearch dumps
src/seekr/eval/
  metrics.py   nDCG / recall with trec_eval conventions
  beir.py      BEIR loader + runner, Anserini reference numbers
bench/         benchmarks and results
docs/          one design note per day
```
