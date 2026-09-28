"""End-to-end crawls against the synthetic web: politeness, robots, dedup, limits, retries."""

from collections import defaultdict

import httpx

from seekr.crawl.crawler import CrawlConfig, Crawler
from seekr.crawl.simweb import SimWeb
from seekr.crawl.store import PageStore


async def _run(web: SimWeb, tmp_path, **cfg):
    log: list[tuple[str, float]] = []
    config = CrawlConfig(seeds=[f"http://{h}/page/0" for h in web.hosts], **cfg)
    store = PageStore(tmp_path / "crawl.db")
    stats = await Crawler(config, store, web.transport(), on_request=lambda u, t: log.append((u, t))).run()
    return stats, store, log


def _min_gap_per_host(log):
    times = defaultdict(list)
    for url, t in log:
        times[url.split("/")[2]].append(t)
    gaps = [b - a for ts in times.values() for a, b in zip(sorted(ts), sorted(ts)[1:], strict=False)]
    return min(gaps) if gaps else float("inf")


async def test_politeness_robots_and_dedup(tmp_path):
    web = SimWeb(n_hosts=4, pages_per_host=25, latency=0.005, seed=3)
    stats, store, log = await _run(web, tmp_path, max_pages=10_000, delay=0.05, concurrency=8, max_depth=20)

    # Never faster than one request per `delay` to the same host.
    assert _min_gap_per_host(log) >= 0.05 - 1e-3
    # Never fetched a robots-disallowed page.
    assert not any("/private/" in u for u, _ in log)
    assert stats.robots_blocked > 0
    # Tracking-parameter variants were normalized, never fetched separately.
    assert not any("utm_source" in u for u, _ in log)

    # Score duplicates as clusters: whichever copy is crawled first is kept, every later member
    # of its cluster must be flagged as a duplicate OF that cluster, and nothing else is flagged.
    rows = store.conn.execute("SELECT url, dup_of FROM pages WHERE status = 200").fetchall()
    members = defaultdict(list)
    for url, _ in rows:
        members[web.cluster(url)].append(url)
    flagged = [(u, d) for u, d in rows if d]
    assert all(web.cluster(u) == web.cluster(d) for u, d in flagged)  # no false matches
    # SimHash is probabilistic: recall is measured (~98.7% at k=8, see bench), precision is exact.
    should_flag = sum(len(m) - 1 for m in members.values())
    assert len(flagged) >= 0.9 * should_flag
    assert stats.exact_dups > 0 and stats.near_dups > 0


async def test_crawl_delay_from_robots_is_honored(tmp_path):
    web = SimWeb(n_hosts=1, pages_per_host=6, latency=0.001, crawl_delay={"site0.test": 0.2}, seed=5)
    _, _, log = await _run(web, tmp_path, max_pages=4, delay=0.01, concurrency=4)
    assert _min_gap_per_host(log) >= 0.2 - 1e-3


async def test_max_pages_and_depth(tmp_path):
    web = SimWeb(n_hosts=3, pages_per_host=40, latency=0.001, seed=9)
    stats, store, _ = await _run(web, tmp_path, max_pages=20, delay=0.0, concurrency=4)
    assert store.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] + stats.robots_blocked <= 20
    stats2, store2, _ = await _run(web, tmp_path / "d0", max_pages=1000, max_depth=0, delay=0.0)
    assert store2.conn.execute("SELECT MAX(depth) FROM pages").fetchone()[0] == 0


async def test_retry_after_on_429_then_success(tmp_path):
    calls = defaultdict(int)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls[request.url.path] += 1
        if calls[request.url.path] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, html="<html><body><p>finally some content here</p></body></html>")

    cfg = CrawlConfig(seeds=["http://slow.test/"], max_pages=1, delay=0.0)
    store = PageStore(tmp_path / "c.db")
    stats = await Crawler(cfg, store, httpx.MockTransport(handler)).run()
    assert stats.retries == 1 and stats.stored == 1 and stats.errors == 0


async def test_server_error_on_robots_blocks_host(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(500)
        return httpx.Response(200, html="<p>should never be fetched</p>")

    cfg = CrawlConfig(seeds=["http://down.test/a"], max_pages=5, delay=0.0, retries=0)
    stats = await Crawler(cfg, PageStore(tmp_path / "c.db"), httpx.MockTransport(handler)).run()
    assert stats.robots_blocked == 1 and stats.fetched == 0


async def test_allowed_hosts_restricts_crawl(tmp_path):
    web = SimWeb(n_hosts=5, pages_per_host=20, latency=0.001, cross_host_ratio=0.5, seed=2)
    config = CrawlConfig(
        seeds=["http://site0.test/page/0"], max_pages=500, delay=0.0, allowed_hosts={"site0.test"}
    )
    store = PageStore(tmp_path / "c.db")
    await Crawler(config, store, web.transport()).run()
    hosts = {u.split("/")[2] for (u,) in store.conn.execute("SELECT url FROM pages")}
    assert hosts == {"site0.test"}


async def test_process_pool_gives_identical_results(tmp_path):
    web = SimWeb(n_hosts=3, pages_per_host=15, latency=0.001, seed=4)
    inline, s1, _ = await _run(web, tmp_path / "a", max_pages=1000, delay=0.0)
    pooled, s2, _ = await _run(web, tmp_path / "b", max_pages=1000, delay=0.0, cpu_workers=2)
    # Crawl order (and so which copy of a duplicate is kept) varies with concurrency, but the
    # fingerprints and the number of duplicates found must not depend on where parsing ran.
    q = "SELECT url, content_hash, simhash FROM pages ORDER BY url"
    assert s1.conn.execute(q).fetchall() == s2.conn.execute(q).fetchall()
    assert inline.exact_dups + inline.near_dups == pooled.exact_dups + pooled.near_dups
