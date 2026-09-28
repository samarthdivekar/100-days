"""The crawl loop: a polite, concurrent, deduplicating breadth-first crawler.

Politeness is the hard constraint, throughput the goal:

* At most ONE request in flight per host, and at least `delay` seconds between requests to the
  same host (raised to the site's Crawl-delay if it asks for more, capped at 30 s).
* robots.txt is fetched once per host and obeyed (RFC 9309, see robots.py).
* 429 and 503 responses are retried with backoff, honoring Retry-After.

Throughput comes from crawling many hosts at once. The scheduler keeps one queue per host and
a "ready" queue of hosts whose politeness delay has elapsed; N workers take the next ready host,
fetch one URL from it, and re-arm that host's timer. So 20 hosts x 1 req/s each gives up to
20 pages/s without ever hitting one server faster than 1 req/s.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import httpx

from seekr.crawl.dedup import NearDupIndex
from seekr.crawl.parse import analyze
from seekr.crawl.robots import RobotsPolicy
from seekr.crawl.store import PageStore
from seekr.crawl.urls import host_of, normalize, path_and_query

DEFAULT_AGENT = "SeekrBot/0.1 (student research crawler; low request rate)"


@dataclass
class CrawlConfig:
    seeds: list[str]
    max_pages: int = 200
    max_depth: int = 5
    concurrency: int = 16
    delay: float = 1.0  # min seconds between requests to one host
    max_delay: float = 30.0  # cap on a site's Crawl-delay
    allowed_hosts: set[str] | None = None  # None = follow links anywhere
    max_bytes: int = 2_000_000
    timeout: float = 15.0
    retries: int = 2
    user_agent: str = DEFAULT_AGENT
    near_dup_bits: int = 8  # SimHash candidate radius, measured (see dedup.py)
    near_dup_jaccard: float = 0.8  # MinHash verification threshold
    cpu_workers: int = 0  # >0: parse + hash in a process pool so CPU work never stalls the event loop


@dataclass
class CrawlStats:
    fetched: int = 0
    stored: int = 0
    errors: int = 0
    non_html: int = 0
    robots_blocked: int = 0
    exact_dups: int = 0
    near_dups: int = 0
    near_dup_rejected: int = 0  # SimHash candidates that failed MinHash verification
    noindex: int = 0
    retries: int = 0
    bytes: int = 0
    started: float = field(default_factory=time.monotonic)
    elapsed: float = 0.0

    @property
    def pages_per_sec(self) -> float:
        return self.fetched / self.elapsed if self.elapsed else 0.0

    @property
    def dup_rate(self) -> float:
        return (self.exact_dups + self.near_dups) / self.fetched if self.fetched else 0.0

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "started"}
        d["pages_per_sec"] = round(self.pages_per_sec, 2)
        d["dup_rate"] = round(self.dup_rate, 4)
        d["elapsed"] = round(self.elapsed, 2)
        return d


@dataclass
class _Host:
    name: str
    queue: deque = field(default_factory=deque)
    robots: RobotsPolicy | None = None
    delay: float = 1.0
    next_time: float = 0.0
    active: bool = False  # in the ready queue, waiting on a timer, or being fetched


class Crawler:
    def __init__(
        self,
        config: CrawlConfig,
        store: PageStore,
        transport: httpx.AsyncBaseTransport | None = None,
        on_request: Callable[[str, float], None] | None = None,
    ):
        self.cfg = config
        self.store = store
        self.transport = transport
        self.on_request = on_request  # (url, monotonic time) hook, used by tests/benchmarks
        self.stats = CrawlStats()
        self.hosts: dict[str, _Host] = {}
        self.seen: set[str] = set()
        self.depth: dict[str, int] = {}
        self.hashes: dict[str, str] = {}
        self.near = NearDupIndex(config.near_dup_bits, config.near_dup_jaccard)
        self._ready: asyncio.Queue[_Host | None] = asyncio.Queue()
        self._active_hosts = 0
        self._scheduled = 0  # URLs admitted to the frontier (bounded by max_pages)
        self._pool: ProcessPoolExecutor | None = None

    # ------------------------------------------------------------------ frontier
    def _admit(self, url: str, depth: int) -> None:
        if url in self.seen or depth > self.cfg.max_depth or self._scheduled >= self.cfg.max_pages:
            return
        host_name = host_of(url)
        if self.cfg.allowed_hosts is not None and host_name not in self.cfg.allowed_hosts:
            return
        self.seen.add(url)
        self.depth[url] = depth
        self._scheduled += 1
        host = self.hosts.get(host_name)
        if host is None:
            host = self.hosts[host_name] = _Host(host_name, delay=self.cfg.delay)
        host.queue.append(url)
        if not host.active:
            host.active = True
            self._active_hosts += 1
            self._arm(host)

    def _arm(self, host: _Host) -> None:
        wait = host.next_time - time.monotonic()
        if wait <= 0:
            self._ready.put_nowait(host)
        else:
            asyncio.get_running_loop().call_later(wait, self._ready.put_nowait, host)

    @staticmethod
    async def _wait_until(deadline: float) -> None:
        # asyncio may wake a timer up to one clock tick early (~15.6 ms on Windows), so re-check
        # the clock instead of trusting a single sleep: politeness must never be violated.
        while (remaining := deadline - time.monotonic()) > 0:
            await asyncio.sleep(remaining)

    def _release(self, host: _Host, requested: bool) -> None:
        if requested:  # a robots-blocked URL made no request, so it doesn't reset the politeness clock
            host.next_time = time.monotonic() + host.delay
        if host.queue:
            self._arm(host)
            return
        host.active = False
        self._active_hosts -= 1
        if self._active_hosts == 0:
            for _ in range(self.cfg.concurrency):
                self._ready.put_nowait(None)

    # ------------------------------------------------------------------ network
    async def _get(self, client: httpx.AsyncClient, url: str) -> httpx.Response | None:
        for attempt in range(self.cfg.retries + 1):
            if self.on_request:
                self.on_request(url, time.monotonic())
            try:
                async with client.stream("GET", url) as resp:
                    if resp.status_code in (429, 503) and attempt < self.cfg.retries:
                        self.stats.retries += 1
                        retry_after = resp.headers.get("retry-after", "")
                        wait = float(retry_after) if retry_after.isdigit() else 2.0 * (attempt + 1)
                        await asyncio.sleep(min(wait, self.cfg.max_delay))
                        continue
                    body = b""
                    ctype = resp.headers.get("content-type", "")
                    if url.endswith("/robots.txt") or "html" in ctype:
                        async for chunk in resp.aiter_bytes():
                            body += chunk
                            if len(body) > self.cfg.max_bytes:
                                break
                    resp._content = body  # keep the (possibly truncated) body for .text
                    self.stats.bytes += len(body)
                    return resp
            except (httpx.HTTPError, OSError):
                if attempt == self.cfg.retries:
                    return None
                await asyncio.sleep(0.5 * (attempt + 1))
        return None

    async def _load_robots(self, client: httpx.AsyncClient, host: _Host, scheme: str) -> None:
        resp = await self._get(client, f"{scheme}://{host.name}/robots.txt")
        host.robots = (
            RobotsPolicy.from_status(resp.status_code, resp.text) if resp else RobotsPolicy.disallow_all()
        )
        asked = host.robots.crawl_delay(self.cfg.user_agent)
        if asked:
            host.delay = max(host.delay, min(asked, self.cfg.max_delay))

    # ------------------------------------------------------------------ one page
    async def _process(self, client: httpx.AsyncClient, host: _Host) -> bool:
        """Fetch one URL from `host`. Returns whether a request was sent to the host."""
        url = host.queue.popleft()
        if host.robots is None:
            await self._load_robots(client, host, url.split("://", 1)[0])
            host.next_time = time.monotonic() + host.delay  # the robots.txt fetch counts too
        if not host.robots.can_fetch(self.cfg.user_agent, path_and_query(url)):
            self.stats.robots_blocked += 1
            return False
        await self._wait_until(host.next_time)
        resp = await self._get(client, url)
        self.stats.fetched += 1
        if resp is None or resp.status_code >= 400:
            self.stats.errors += 1
            self.store.save_page(
                url=url,
                final_url=url,
                status=resp.status_code if resp else 0,
                depth=self.depth[url],
                page=None,
                content_hash=None,
                simhash=None,
                dup_of=None,
                dup_kind=None,
            )
            return True
        if "html" not in resp.headers.get("content-type", ""):
            self.stats.non_html += 1
            return True
        final_url = normalize(str(resp.url)) or url
        if self._pool is not None:
            loop = asyncio.get_running_loop()
            page, chash, fp, sig = await loop.run_in_executor(self._pool, analyze, resp.text, final_url)
        else:
            page, chash, fp, sig = analyze(resp.text, final_url)
        dup_of, dup_kind = None, None
        if chash in self.hashes:
            dup_of, dup_kind = self.hashes[chash], "exact"
            self.stats.exact_dups += 1
        elif (near := self.near.find(fp, sig)) is not None:
            dup_of, dup_kind = near, "near"
            self.stats.near_dups += 1
        else:
            self.hashes[chash] = url
            self.near.add(fp, url, sig)
        self.stats.noindex += page.noindex
        self.store.save_page(
            url=url,
            final_url=final_url,
            status=resp.status_code,
            depth=self.depth[url],
            page=page,
            content_hash=chash,
            simhash=fp,
            dup_of=dup_of,
            dup_kind=dup_kind,
            html=resp.text,
        )
        self.stats.stored += 1

        # A duplicate's links were already discovered on the original; nofollow pages give none.
        if dup_kind is None and not page.nofollow:
            for link in page.links:
                self._admit(link, self.depth[url] + 1)
            if final_url != url:
                self.seen.add(final_url)
        return True

    async def _worker(self, client: httpx.AsyncClient) -> None:
        while True:
            host = await self._ready.get()
            if host is None:
                return
            requested = True
            try:
                requested = await self._process(client, host)
            finally:
                self._release(host, requested)

    async def run(self) -> CrawlStats:
        for seed in self.cfg.seeds:
            url = normalize(seed)
            if url:
                self._admit(url, 0)
        if self._active_hosts == 0:
            return self.stats
        headers = {"User-Agent": self.cfg.user_agent, "Accept": "text/html,application/xhtml+xml"}
        if self.cfg.cpu_workers > 0:
            self._pool = ProcessPoolExecutor(max_workers=self.cfg.cpu_workers)
        try:
            await self._crawl(headers)
        finally:
            if self._pool is not None:
                self._pool.shutdown(cancel_futures=True)
                self._pool = None
        self.stats.near_dup_rejected = self.near.rejected
        self.stats.elapsed = time.monotonic() - self.stats.started
        self.store.save_run(self.stats.as_dict())
        return self.stats

    async def _crawl(self, headers: dict) -> None:
        async with httpx.AsyncClient(
            headers=headers,
            timeout=self.cfg.timeout,
            follow_redirects=True,
            transport=self.transport,
            limits=httpx.Limits(max_connections=self.cfg.concurrency * 2),
        ) as client:
            workers = [asyncio.create_task(self._worker(client)) for _ in range(self.cfg.concurrency)]
            await asyncio.gather(*workers)


def crawl(config: CrawlConfig, store: PageStore, transport=None, on_request=None) -> CrawlStats:
    return asyncio.run(Crawler(config, store, transport, on_request).run())
