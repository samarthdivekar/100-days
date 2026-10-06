# 100 Days, 100 Projects

One project a day from Sep 28, 2026 to Jan 5, 2027. Ten tracks of ten days; each day ships a
standalone project, and each track's ten projects combine into one flagship system on its tenth
day. Every project has tests, a design note, and one number that shows it works.

**Progress: 3 / 100**

| # | Track | Days | Flagship | Status |
|---|---|---|---|---|
| 1 | [Search engine from scratch](track01-seekr/) | 1–10 · Sep 28 – Oct 7 | Seekr | 🟢 in progress (3/10) |
| 2 | Distributed systems & storage | 11–20 · Oct 8 – 17 | RaftKV | ⏳ |
| 3 | Recommendation & ranking | 21–30 · Oct 18 – 27 | ShopRec | ⏳ |
| 4 | LLM systems & inference | 31–40 · Oct 28 – Nov 6 | Clearance v2 | ⏳ |
| 5 | Security engineering | 41–50 · Nov 7 – 16 | SOC Copilot | ⏳ |
| 6 | Trust & safety / integrity ML | 51–60 · Nov 17 – 26 | IntegrityHub | ⏳ |
| 7 | Data engineering & MLOps | 61–70 · Nov 27 – Dec 6 | FraudOps | ⏳ |
| 8 | Geospatial, disaster & nature AI | 71–80 · Dec 7 – 16 | ReliefMap | ⏳ |
| 9 | On-device, mobile & privacy ML | 81–90 · Dec 17 – 26 | PrivateWallet | ⏳ |
| 10 | Autonomous systems, RL & optimization | 91–100 · Dec 27 – Jan 5 | 100-Day Report | ⏳ |

## Log

| Day | Date | Project | Result |
|---|---|---|---|
| 3 | Oct 6 | [BM25 ranking, checked against Anserini](track01-seekr/docs/day03-bm25.md) | Within 0.0045 nDCG@10 of Anserini on 5 BEIR datasets; on Wikipedia the right article ranks first 88% of the time (unranked: 15%) |
| 2 | Sep 29 | [Compressed positional inverted index](track01-seekr/docs/day02-index.md) | All 278k Simple Wikipedia articles indexed in 8 min; 3.2× compression; single-term p50 1.5 ms, phrase p50 60 ms |
| 1 | Sep 28 | [Polite, deduplicating web crawler](track01-seekr/docs/day01-crawler.md) | 55 pages/s (82.5% of the politeness ceiling), 0 violations; near-duplicate false positives on a real site cut from 25 to 0 |

## Repository layout

```
track01-seekr/     one folder per track: a Python package, tests, benchmarks, docs/dayNN-*.md
...
```

Each track installs on its own: `pip install -e "trackNN-name[dev]"`, then `python -m pytest`
inside the folder.
