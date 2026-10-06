# Day 3: BM25 ranking, checked against Anserini

**Result:** a from-scratch BM25 that lands within **0.0045 nDCG@10** of Anserini's published BM25 on
all five BEIR datasets tested (mean gap 0.0017), with recall@1000 within 0.003.
On all 278k Simple English Wikipedia articles, searching an article's title puts that article
**first 88.2% of the time** (Day 2's unranked results: 15.4%), at p50 3 ms.

## What changed

| Piece | Decision | Why |
|---|---|---|
| Scoring | BM25, Lucene's formula: `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`, `tf / (tf + k1·(1 − b + b·|d|/avgdl))`, query terms weighted by their count | To compare with Anserini, it has to be the same function, not "a BM25" |
| Analyzer | `english`: drop possessive `'s`, Lucene's 33 English stopwords, Porter stemmer (Snowball implementation of the original algorithm). Original positions kept, so phrases still need the gap a stopword left | Same pipeline as Lucene's EnglishAnalyzer, which Anserini uses; the analyzer name is stored in the index so queries are always analyzed the same way |
| Evaluation | nDCG@10, R@100, R@1000 with trec_eval conventions (ties broken by doc id descending, every judged query counts, gain = grade); a query never retrieves its own id (Anserini `-removeQuery`) | Small conventions move scores by more than the differences being measured |
| Execution | Term-at-a-time into a dense float32 accumulator, `argpartition` for top-k | One vectorized update per query term; p50 under 10 ms on 171k documents |
| Search UX | Plain words rank the whole corpus; phrases, `-exclusions` and `OR` decide which documents qualify, BM25 orders them | Ranking without losing Day 2's exact-match operators |
| Crawled pages | Indexed from the stored HTML's *main content* | Day 2 indexed menus: on books.toscrape.com, `poetry` matched all 38 pages through the sidebar |

## BEIR results (nDCG@10)

Anserini's numbers are its BM25 "flat" regressions (title + text in one field, k1 = 0.9, b = 0.4),
from `castorini/anserini` `docs/reproduce/from-document-collection/beir-v1.0.0-<name>.flat.md`.

| Dataset | Docs | Queries | A: plain tokens | B: English analyzer | **C: + Lucene length norms** | Anserini | C − Anserini |
|---|---|---|---|---|---|---|---|
| SciFact | 5,183 | 300 | 0.6633 | 0.6769 | **0.6781** | 0.6789 | −0.0008 |
| NFCorpus | 3,633 | 323 | 0.3043 | 0.3205 | **0.3203** | 0.3218 | −0.0015 |
| ArguAna | 8,674 | 1,406 | 0.3802 | 0.3997 | **0.3973** | 0.3970 | +0.0003 |
| FiQA | 57,638 | 648 | 0.2307 | 0.2362 | **0.2348** | 0.2361 | −0.0013 |
| TREC-COVID | 171,332 | 50 | 0.5787 | 0.5931 | **0.5902** | 0.5947 | −0.0045 |
| **Mean** | | | 0.4314 | 0.4453 | **0.4441** | 0.4457 | −0.0016 |

Recall agrees as closely: R@1000 for C vs Anserini is 0.9767/0.9767 (SciFact), 0.3701/0.3704,
0.9879/0.9872, 0.7393/0.7393 and 0.3938/0.3955. Full numbers: `bench/results/day03_beir.json`.

**What the columns show**

- **Stemming and stopwords are worth +0.014 nDCG on average** (A → B), positive on every dataset.
- **Lucene's 1-byte length norms didn't close the gap.** Lucene stores each document length in one byte
  (exact below 24, then 4 significant bits). I expected copying that to explain the residual difference;
  it didn't: mean |gap| is 0.0015 with exact lengths and 0.0017 with quantized ones. The remaining
  differences most likely come from tokenization (Lucene's StandardTokenizer treats hyphens, numbers
  and URLs differently from a letter/digit splitter).
- **Ties are real under quantization.** With quantized lengths, many documents share a score exactly,
  so tie-breaking conventions matter; the tests compare rankings by score, not by id order.

## Parameters

| Dataset | k1 = 0.9, b = 0.4 (Anserini) | k1 = 1.2, b = 0.75 (textbook) |
|---|---|---|
| SciFact | 0.6781 | 0.6791 |
| NFCorpus | 0.3203 | 0.3210 |
| ArguAna | 0.3973 | 0.4785 |
| FiQA | 0.2348 | 0.2501 |
| TREC-COVID | 0.5902 | 0.6094 |
| Mean | 0.4441 | 0.4676 |

The textbook defaults score higher on all five, mostly through ArguAna (long argumentative queries
favour stronger length normalization). These were chosen *after* seeing test results, so this is a
sensitivity check, not a fair improvement over Anserini. Seekr's product search uses k1 = 1.2,
b = 0.75; `BM25()` itself defaults to Anserini's values so the reproduction stays reproducible.

## Latency (BEIR, config C)

Term-at-a-time over a dense accumulator: p50 3.1 ms (SciFact), 4.2 ms (FiQA, 58k docs),
9.6 ms (TREC-COVID, 171k docs); ArguAna's ~200-word queries take p50 25 ms / p99 63 ms because every
query word is scored. Measured on a machine also running two long ML experiments.

## Seekr on Wikipedia

The product index was rebuilt with the English analyzer: 278,214 articles, 51.0M indexed terms
(stopwords no longer stored), 992k distinct terms, postings 136.6 MB (Day 2: 168.2 MB), built in
180 s with 4 worker processes.

Wikipedia has no relevance judgments, so this is a **known-item** test: 500 sampled article titles
(two or more words) as queries; the one right answer is the article itself.

| | BM25 (k1 = 1.2, b = 0.75) | Day 2: all matches, index order |
|---|---|---|
| Right article ranked first | **88.2%** | 15.4% |
| Right article in top 10 | **98.6%** | 67.4% |
| MRR@10 | **0.925** | 0.313 |
| Latency p50 / p95 / p99 | 3.0 / 11.1 / 36.0 ms | 1.0 / 5.2 / 26.6 ms |

Examples: `New York Giants` → *New York Giants*, *Polo Grounds*, *2012 New York Giants season*;
`Schloss Johannisberg` → *Schloss Johannisberg*, *Schloss Ettersburg*, *Neuschwanstein Castle*.

Titles are an easy query type (the words are in the article, and in its title); this measures
whether ranking puts the obvious answer first, not hard relevance. BEIR above is the hard test.

**The crawl fix on real pages.** Re-indexing the 38 crawled books.toscrape.com pages from their main
content: `poetry` now matches 1 page (the Poetry category) instead of all 38, and indexed terms
dropped from 9,618 to 4,760.

## Limits

- Exhaustive term-at-a-time scoring touches every posting of every query term; dynamic pruning
  (MaxScore / WAND) would skip documents that cannot reach the top k. The dense accumulator also
  costs O(N) memory per query.
- One text field: no title boost or field weighting yet (Anserini's "multifield" variant).
- Parameters are global; no per-collection tuning on held-out queries.

## Try it

```bash
python bench/bench_beir.py --workers 3          # needs data/beir/<name> (BEIR zips from public.ukp.informatik.tu-darmstadt.de)
python -m seekr.cli index --cirrus data/wiki/simplewiki-20251229-cirrussearch-content.json.gz --out data/index-en --analyzer english --workers 4
python -m seekr.cli search "black holes in galaxies" --index data/index-en
python bench/bench_wiki_rank.py --index data/index-en
```
