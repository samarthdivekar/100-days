# Day 2: a compressed, positional inverted index

**Result:** all 278,214 Simple English Wikipedia articles (67.8M tokens) indexed in 8 minutes; postings are
3.2× smaller than 32-bit integers and 41% the size of the raw text while keeping every word position;
single-term p50 1.5 ms, two-term AND p50 6.9 ms, phrase p50 60–71 ms; every query type matches a brute-force
scan, and 1,000 full-scale phrase queries all returned their source article.

## What it does

```
Wikipedia dump / Day 1 crawl
        │
        ▼
 batches of ~1M tokens ──(process pool)──▶ sorted, VByte-compressed block files      (SPIMI)
        │                                             │
        ▼                                             ▼
 doc table (SQLite: url, title, length,     k-way merge by term ──▶ postings.bin + lexicon (SQLite)
 compressed text)                                          │
                                                           ▼
                            queries: AND / OR / NOT / "phrase" over memory-mapped postings
```

| Piece | Decision | Why |
|---|---|---|
| Tokens | NFKC-normalized, case-folded letter/digit runs; every position kept | Phrase queries need positions; stemming waits for Day 3, where its effect on ranking can be measured |
| Postings layout | Per term: doc-id gaps, term frequencies, then per-doc position gaps, contiguous in `postings.bin` | One sequential read per term; tf and doc lengths are what BM25 needs tomorrow |
| Compression | VByte on gaps | Byte-aligned and simple; gaps are small, so most take 1 byte |
| VByte implementation | Pure Python below ~1,000 values, NumPy above | Most terms are rare (Zipf); NumPy costs ~100 µs per call even for one integer |
| Build | SPIMI: fixed-size batches → sorted blocks → k-way merge; blocks built in a process pool | Memory stays bounded whatever the corpus size; blocks are independent (map), the merge is the reduce |
| Determinism | Batch boundaries depend only on the input; blocks merge in batch order | Parallel and serial builds are byte-identical, and a test enforces it |
| AND | Rarest list first, binary-search (`searchsorted`) into longer lists | O(m log n) instead of O(m + n) when lists are very different in length |
| Phrase | Candidates = AND of the words; then one vectorized intersection over `doc << 32 \| position` keys, each shifted by its offset in the phrase | No per-document Python loop |
| Correctness | Every query type checked against a brute-force scan of the documents (303 random queries); multi-block build byte-identical to single-block; parallel byte-identical to serial | An index that is fast and wrong is worthless |

## Corpus

Simple English Wikipedia, CirrusSearch content dump of 2025-12-29 (636 MB gzipped). It carries
plain article text, so no lossy wikitext cleaning is needed. Downloaded from dumps.wikimedia.org,
which is what Wikimedia asks bulk users to do instead of crawling (Day 1's crawler received a 403).

## Size

| | |
|---|---|
| Articles / tokens / distinct terms | 278,214 / 67.8M / 1.09M |
| Postings (doc, term) pairs | 34.2M |
| Raw article text | 408.6 MB |
| **Postings file** | **168.2 MB** (doc ids 45.1 · term frequencies 34.2 · positions 89.0) |
| Same data as fixed 32-bit integers | 544.4 MB → **3.24× compression** |
| Bits per position / per doc id | 10.5 / 10.6 |
| Skip tables (3,885 terms with df ≥ 1,024) | 0.8 MB |
| Build: 8 worker processes | 483 s total, of which merge 114.5 s |

## Query latency

500 seeded queries per type, sampled from the articles themselves (so each has a known answer).
Same query sets in every column. Machine shared with a background CPU-bound job, so absolute numbers
are pessimistic; the columns are comparable with each other.

| Query | First version p50 / p99 | + faster VByte decode | **+ position skip tables** |
|---|---|---|---|
| Single term | 3.9 / 43 ms | 1.7 / 9.3 ms | **1.5 / 8.6 ms** |
| Two-term AND | 21.6 / 96 ms | 7.0 / 38.6 ms | **6.9 / 40.0 ms** |
| 2-word phrase | 126 / 1,283 ms | 80 / 1,146 ms | **60 / 1,164 ms** |
| 3-word phrase | 298 / 1,441 ms | 156 / 801 ms | **71 / 1,107 ms** |

Correctness at full scale: 0 of 1,000 phrase queries failed to return the article they were taken from.

**The tail is common-word phrases.** Every one of the slowest queries pairs two very frequent
words: `"of the"` (138,189 matching articles, 1.8 s), `"in the"`, `"is the"`, `"at the"`, `"the united"`.
Both words occur almost everywhere, so skip tables cannot help and the result set itself is huge.
The standard fix is a separate index of frequent word pairs (Williams, Zobel & Bahle, *Fast phrase
querying with combined indexes*, 2004); it is on the list for the Day 10 serving work.

## Performance debugging log

**1. NumPy per-call overhead.** The first build took minutes per 4M-token block. Profiling showed the
vectorized VByte encoder costing ~100 µs per call *even for one integer*. Term frequencies follow
Zipf's law, so most postings lists hold one or two documents: hundreds of thousands of tiny calls per
block. Measured crossovers (encode ~1,000 values, decode ~500 bytes) now pick pure Python for short
lists and NumPy for long ones.

| n values | encode: Python / NumPy | decode: Python / NumPy |
|---|---|---|
| 1 | 0.7 / 103 µs | 0.7 / 98 µs |
| 128 | 47 / 317 µs | 141 / 281 µs |
| 8,192 | 7,268 / 3,103 µs | 11,550 / 4,875 µs |

**2. A background process was eating the CPU.** `list.append` was measuring 2 µs instead of ~30 ns.
A CPU-bound job from another project had been running for 26 hours on all cores. Lowering it to idle
priority halved build time (132 s → 64 s for 8,000 articles).

**3. Parallel SPIMI.** Batches are indexed into blocks by a process pool and merged in batch order,
so parallel output is byte-identical to serial (tested). On 8,000 articles: 75.8 s serial → 42.0 s with
6 workers (1.8×). The single-threaded merge is now 17 of those 42 seconds (Amdahl's law); on the full
dump it is 115 of 483 s.

**4. Faster VByte decode.** Decoding the 3.4M positions of "the" took 290 ms at 84 ns per value,
though nearly every value is one byte. A decoder that loops over byte *length* (at most 3 passes) with
an all-one-byte fast path is 3–24× faster on real postings (doc ids of "the": 28 → 1.2 ms). A boolean
doc-id table replaced a sort-based `np.isin` in phrase filtering.

**5. Skip tables.** For terms with df ≥ 1,024, the byte offset of every 128th document's position run
is recorded. A phrase decodes only the runs holding candidate documents; because every run starts with
an absolute position, the runs are concatenated and decoded in one call. The tables are computed from
`postings.bin` alone, so the existing index gained them in 1.9 s without a rebuild.

## Limits

- The lexicon lives in SQLite (one indexed lookup per query term, ~tens of µs); a production
  engine would keep a front-coded, memory-resident dictionary.
- Skip tables cover positions only; an AND still decodes each doc-id list in full before intersecting.
- Phrases of two common words ("of the") take 1–2 s; a pair index would fix this.
- The merge is single-threaded, so it caps the parallel speedup (Amdahl's law).

## Try it

```bash
python -m seekr.cli index --cirrus data/wiki/simplewiki-20251229-cirrussearch-content.json.gz --out data/index --workers 8
python -m seekr.cli search '"solar system" planet -dwarf' --index data/index
python bench/bench_query.py --index data/index
python bench/bench_index_build.py --dump data/wiki/simplewiki-20251229-cirrussearch-content.json.gz --docs 8000 --workers 0 6
```
