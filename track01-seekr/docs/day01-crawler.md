# Day 1: a polite, deduplicating web crawler

**Result:** 55 pages/s on a 20-host synthetic web (82.5% of the politeness ceiling), zero robots.txt
or politeness violations, duplicate precision 1.00 / recall 0.986. On a real site, 2 of 2 true
duplicates found with 0 false positives, after two measured fixes described below.

## What it does

```
seeds ─▶ normalize URL ─▶ per-host queue ─▶ ready queue (politeness timers) ─▶ N async workers
                                                                                   │
                     robots.txt (RFC 9309, once per host) ◀────────────────────────┤
                                                                                   ▼
   links ◀── parse HTML (text, main content, links, meta robots, canonical) ◀── fetch (retry/backoff)
     │                                  │
     ▼                                  ▼
  frontier                   exact hash ─▶ SimHash candidates ─▶ MinHash verify ─▶ SQLite (+ compressed HTML)
```

| Concern | Decision | Why |
|---|---|---|
| Politeness | ≤ 1 request in flight per host, ≥ `delay` s between requests, raised to the site's `Crawl-delay` (capped at 30 s); the clock is re-checked before every request | asyncio timers can fire one clock tick (~15.6 ms on Windows) early; a test caught a 47 ms gap under a 50 ms delay |
| Throughput | One queue per host + a ready queue of hosts; workers take whichever host is due | Many hosts in parallel, each one slow; the ceiling is `hosts / (delay + latency)` |
| robots.txt | Own parser, RFC 9309: longest match wins, `allow` wins ties, `*` and `$` wildcards, 4xx → allow all, 5xx → disallow all | Python's `urllib.robotparser` applies rules in file order and ignores wildcards |
| URL identity | Lowercase scheme/host, default ports dropped, dot segments resolved, percent-encoding canonicalized, query sorted, `utm_*`/`gclid`/`fbclid` removed, fragment dropped | Otherwise every tracking link is a "new" page |
| Retries | 429/503 retried with backoff, honoring `Retry-After` | Back off when a server asks |
| Storage | SQLite: pages, link graph (for Day 4 PageRank), zlib-compressed HTML | Reprocess without refetching (used below) |
| CPU work | Parse + hash can run in a process pool (`--cpu-workers`) | Off by default: after optimization, pickling cost more than the 0.4 ms of work |

## Benchmark (synthetic web with ground truth)

`python bench/bench_crawler.py`: 20 hosts × 100 pages, 50 ms latency, 0.25 s politeness delay,
~8% exact and ~8% near duplicates, robots-disallowed `/private/` links, tracking-parameter variants.

| Workers | Pages/s | % of ceiling | Dup precision | Dup recall | robots violations | politeness violations |
|---|---|---|---|---|---|---|
| 1 | 13.3 | 19.9% | 1.00 | 0.986 | 0 | 0 |
| **32** | **55.0** | **82.5%** | **1.00** | **0.986** | **0** | **0** |
| 32 + 4 parse processes | 55.1 | 82.6% | 1.00 | 0.986 | 0 | 0 |

### Throughput: 36 → 59 pages/s

The first run plateaued at 36 pages/s (54% of the ceiling) whether 4 or 32 workers ran, so more
concurrency was not the fix. Per-host timelines showed every host continuously busy, but with a
median of 0.55 s between requests instead of the ideal 0.30 s. Profiling: SimHash took 12.6 s of
a 23 s crawl, blocking the single event loop so timers fired late.

Rewriting SimHash to hash each unique word once (cached) and combine word hashes into shingle
hashes with NumPy made it **14× faster** (5.54 → 0.40 ms per page) with the same accuracy, and
throughput rose to 59 pages/s (88%). The final pipeline adds main-content extraction and MinHash,
settling at 55 pages/s.

## Duplicate detection: three measured iterations

**1. The paper's threshold doesn't transfer.** Google's SimHash paper (Manku et al., WWW 2007) used
Hamming distance ≤ 3. On our pages it caught only **55%** of near-duplicates (one edited word plus
a new timestamp). Unrelated pages were never closer than 17 bits in 20,000 random pairs.

| k | near-dup recall | false positives (20k unrelated pairs) |
|---|---|---|
| 3 | 0.47 | 0 |
| 6 | 0.92 | 0 |
| **8** | **0.99** | **0** |
| 10 | 1.00 | 0 |

**2. Real sites are templates.** A polite crawl of `books.toscrape.com` (a sandbox built for crawler
practice; 200 pages at 1 req/s) flagged **27 of 200** pages as near-duplicates. Checking word overlap
showed most were category pages listing *different* books that shared a 50-link sidebar.
Fingerprinting only the main content (dropping `<nav>`, `<header>`, `<footer>`, `<aside>` and
elements whose class/id says nav, menu, sidebar, breadcrumb...) cut that to 19, still mostly wrong:
every listing repeats "In stock · Add to basket · £ · ★" 20 times, and those repeats outvoted the
titles.

**3. Count shingles once, and verify.** Features became the *set* of word 3-shingles, and a SimHash
candidate is only accepted if a 64-value MinHash signature estimates the shingle Jaccard at ≥ 0.8.
Scored against exact Jaccard over all 19,900 page pairs, replaying the stored HTML in crawl order
(no re-crawl needed):

| Pipeline | Flagged | Correct | False |
|---|---|---|---|
| Full text, SimHash k=8 | 27 | 2 | 25 |
| Main content, SimHash k=8 | 19 | 2 | 17 |
| Main content, shingle *set*, SimHash k=8 | 3 | 2 | 1 |
| **+ MinHash verification ≥ 0.8** | **2** | **2** | **0** |

The site's only true duplicate cluster is `/`, `/index.html` and `/catalogue/category/books_1/`.

## Real-world notes

- **Wikipedia returned 403** ("Please respect our robot policy"): Wikimedia blocks crawlers whose
  User-Agent has no contact information. The crawler recorded the error and moved on; it does not
  disguise itself as a browser. Use `--contact <your repo URL>` for small crawls, and the Wikipedia
  dumps for the Day 10 corpus, as Wikimedia asks of bulk users.
- On `books.toscrape.com`, throughput (0.6 pages/s) is set by the 1 s politeness delay plus the
  server's response time, as intended.

## Limits

- The frontier and duplicate index live in memory: fine for ~10⁶ pages, not for a web-scale crawl
  (that needs a disk-backed frontier and a sharded fingerprint store).
- MinHash signatures cost 256 bytes per page in memory.
- Main-content extraction is heuristic (tags + class names), not a learned boilerplate model.
- No JavaScript rendering; pages built client-side look empty.

## Try it

```bash
pip install -e ".[dev]"
python -m pytest
python bench/bench_crawler.py
python -m seekr.cli crawl https://books.toscrape.com/ --max-pages 200 --delay 1.0 --db data/books.db
python -m seekr.cli crawl-stats --db data/books.db
```
