# Day 4: PageRank, and whether it beats counting links

**Result:** PageRank over Simple English Wikipedia's 12.6M links converges in 28 iterations (4.6 s)
and matches networkx to 1e-9. As a ranking prior it lifts "right article ranked first" on
realistic query traffic from **72.8% to 81.4%** (+8.6 points, 95% CI 6.2–11.0, p ≈ 4×10⁻¹²) without
hurting uniformly sampled queries. **But plain in-degree does just as well** (81.2%; PageRank vs
in-degree p = 0.89). PageRank does predict what people read better (Spearman 0.354 vs 0.294 with
pageviews), and its top is polluted by citation-template links.

## What it does

```
CirrusSearch dump ──pass 1──▶ doc ids (same filter as the index) + title & redirect-alias map
                  ──pass 2──▶ outgoing links resolved to doc ids ──▶ edge list (src, dst)
                                                                        │
                                   power iteration: gather r[src]/outdeg, bincount into dst
                                                                        ▼
                         pagerank.npy  ──▶  search score = BM25 + w · log(PageRank · N)
```

| Piece | Decision | Why |
|---|---|---|
| Graph | Two streaming passes over the dump; links resolved through 111,216 redirect aliases; duplicate links and self-links dropped | Links often target a redirect ("USA"); without aliases they'd vanish. Two passes keep 18M link strings out of memory |
| Alignment | Doc ids assigned with exactly the index's filter; the run refuses to save unless all 278,214 titles match the index | A prior that is off by one document is silently wrong for every query |
| PageRank | Power iteration on an edge list: gather `r[src]/outdeg[src]`, scatter-add with `np.bincount`; dangling pages' rank spread evenly; stop when L1 change < N·10⁻¹⁰ | No sparse-matrix library; one iteration on 12.6M edges takes 165 ms |
| Correctness | Matches `networkx.pagerank` to 1e-9 on 6 random graphs (two damping factors, 20% dangling pages); ranks sum to 1 | An independent reference, not a self-check |
| Ranking prior | `BM25 + w · log(PageRank · N)`; `w` chosen on a training query set, reported on disjoint test sets | Day 3's parameter comparison looked at test results; this one doesn't |

## The link graph

| | |
|---|---|
| Articles | 278,214 |
| Link entries in the dump | 18,269,404 |
| **Edges after resolving** | **12,623,362** |
| Links to pages outside the corpus (red links, other namespaces) | 5,459,029 (30%) |
| Redirect aliases used | 111,216 |
| Self-links dropped | 8,224 |
| Dangling pages (no out-links) | 1,651 |
| Graph build (2 passes over 636 MB gzip) | 113 s |
| PageRank | 28 iterations, 4.6 s (165 ms/iteration), converged |

## Does PageRank measure importance?

Validated against two signals Wikimedia publishes in the dump: its own incoming-link counts, and
`popularity_score`, a share of pageviews, i.e. what readers actually open.

| Spearman rank correlation | |
|---|---|
| PageRank vs pageviews | **0.354** |
| In-degree vs pageviews | 0.294 |
| Wikipedia's incoming_links vs pageviews | 0.288 |
| PageRank vs in-degree | 0.885 |
| Our in-degree vs Wikipedia's incoming_links | 0.713 |

Top-1,000 overlap with the most-viewed pages: PageRank 16.5%, in-degree 10.4%.

**Two findings:**

1. **Citation templates pollute the top.** The highest PageRank article is *Wayback Machine*,
   followed by *United States*, *International Standard Book Number*, *France*, *Geographic coordinate
   system*, *Internet Movie Database*, *United Kingdom* and *Digital Object Identifier*. Citation and
   infobox templates add those links to thousands of pages automatically: the link-graph version of
   Day 1's repeated-sidebar problem. The dump records links per page, not which template produced them,
   so they can't simply be filtered out here.
2. **Attention follows culture, not links.** The most-viewed pages include *Black*, *XXXTentacion*,
   *Taylor Swift*, *Google* and *Kobe Bryant*. Link structure can only partly predict that, which caps
   any link signal's correlation with pageviews around the values above.

## Does it make search better?

Known-item test: query = an article's title, the right answer = that article.

* **Traffic** queries sample articles in proportion to pageviews, approximating what people search for.
* **Uniform** queries sample every article equally, the worst case for a popularity prior.
* The prior weight `w` was chosen on 400 separate *training* traffic queries; test sets are disjoint
  (1,000 queries each).

| Prior | Tuned w | Traffic: right article first | MRR@10 | Single-word traffic queries | Uniform: right article first |
|---|---|---|---|---|---|
| None (Day 3) | – | 72.8% | 0.811 | 64.0% | 86.6% |
| In-degree, log(1 + links in) | 0.2 | 81.2% | 0.876 | 78.3% | 86.5% |
| **PageRank, log(PR · N)** | 0.3 | **81.4%** | **0.877** | **79.6%** | 86.4% |

Paired exact McNemar tests on the traffic test set:

| Comparison | Queries fixed | Queries broken | p |
|---|---|---|---|
| None → PageRank | 122 | 36 | 3.9×10⁻¹² |
| None → in-degree | 112 | 28 | 4.2×10⁻¹³ |
| In-degree → PageRank | 25 | 23 | **0.89** |

Bootstrap 95% CI of PageRank's gain over no prior: **+6.2 to +11.0 points**.

**Conclusion:** a link-based prior is clearly worth having, especially for short ambiguous queries
(+15.6 points on single words), and it costs nothing on uniformly sampled ones. For this task,
PageRank's extra machinery is **not** distinguishable from counting incoming links. Seekr uses
PageRank as the default (it predicts attention better and ties on search), but the honest summary is
"use a link prior; in-degree is nearly as good and simpler."

### Examples (`seekr search`, default settings)

| Query | BM25 only: #1 | BM25 + PageRank: top 3 |
|---|---|---|
| `Paris` | *Hi! PARIS* | **Paris**, Île-de-France, University of Paris |
| `Jordan` | *Jordan River (disambiguation)* | **Jordan**, Michael Jordan, Jordan River |
| `Mercury` | *Mercury* (disambiguation page) | Mercury (element), Mercury Records, Mercury (planet) |

The last row is a judgment call: the disambiguation page is a reasonable answer to a bare `Mercury`.

## Limits

- Template-generated links inflate a handful of "infrastructure" pages; filtering them needs
  per-link template provenance, which the dump doesn't carry.
- The prior is global. Personalized or topic-sensitive PageRank (Haveliwala, 2002) would rank
  *Mercury (planet)* above *Mercury Records* for a space-themed session.
- Known-item title queries measure navigational search, not topical relevance; BEIR (Day 3) has no
  link graph to test a prior on.
- Spam resistance (link farms) is untested; Wikipedia's graph is curated.

## Try it

```bash
python -m seekr.cli links --cirrus data/wiki/simplewiki-20251229-cirrussearch-content.json.gz --index data/index-en
python -m seekr.cli search "Jordan" --index data/index-en               # BM25 + PageRank prior
python -m seekr.cli search "Jordan" --index data/index-en --prior none   # BM25 only
python bench/bench_pagerank.py --dump data/wiki/simplewiki-20251229-cirrussearch-content.json.gz --index data/index-en
python bench/bench_prior.py --index data/index-en
```
