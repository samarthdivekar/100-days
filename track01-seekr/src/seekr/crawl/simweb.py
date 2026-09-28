"""A synthetic web with known ground truth, served through httpx.MockTransport.

Real sites can't tell you what the right answer was, and benchmarking against them means
hammering someone's server. This fake web knows exactly which pages are exact duplicates,
near-duplicates (same article, new timestamp and one edited word), robots-disallowed, or the
same page behind a tracking parameter, so the crawler's behavior can be scored precisely.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import httpx

# Every page carries the same site chrome, like real templated sites do.
_CHROME = (
    '<header class="masthead">SimShop home books music travel</header><aside class="side_categories"><ul>'
    + "".join(
        f"<li>category {name}</li>"
        for name in "travel mystery history poetry science fantasy "
        "romance horror humor sports music art cooking health business religion".split()
    )
    + "</ul></aside>"
)
_FOOTER = "<footer>Copyright SimShop. Prices and availability may change. Terms privacy contact.</footer>"


@dataclass
class SimWeb:
    n_hosts: int = 10
    pages_per_host: int = 60
    links_per_page: int = 8
    cross_host_ratio: float = 0.3
    exact_dup_rate: float = 0.08
    near_dup_rate: float = 0.08
    private_pages: int = 5  # per host, disallowed by robots.txt
    latency: float = 0.02  # seconds per response
    crawl_delay: dict[str, float] = field(default_factory=dict)  # host -> Crawl-delay
    seed: int = 7

    def __post_init__(self):
        rng = random.Random(self.seed)
        letters = "abcdefghijklmnopqrstuvwxyz"
        vocab = ["".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(4000)]
        self.hosts = [f"site{i}.test" for i in range(self.n_hosts)]
        self.pages: dict[str, str] = {}  # url -> body text
        self.exact_dups: set[str] = set()
        self.near_dups: set[str] = set()
        self.private: set[str] = set()
        self.source: dict[str, str] = {}  # copy -> the original it was made from
        urls = [f"http://{h}/page/{j}" for h in self.hosts for j in range(self.pages_per_host)]
        originals: list[str] = []
        for url in urls:
            roll = rng.random()
            if originals and roll < self.exact_dup_rate:
                src = rng.choice(originals)
                self.pages[url] = self.pages[src]
                self.exact_dups.add(url)
                self.source[url] = src
            elif originals and roll < self.exact_dup_rate + self.near_dup_rate:
                src = rng.choice(originals)
                self.source[url] = src
                words = self.pages[src].split(" ")
                words[rng.randrange(len(words))] = rng.choice(vocab)  # one edited word
                stamp = f"updated {rng.randint(1, 28)} october 2026 at {rng.randint(0, 23)}h"
                self.pages[url] = " ".join(words) + " " + stamp
                self.near_dups.add(url)
            else:
                self.pages[url] = " ".join(rng.choice(vocab) for _ in range(rng.randint(250, 400)))
                originals.append(url)
        for h in self.hosts:
            for j in range(self.private_pages):
                self.private.add(f"http://{h}/private/{j}")
        # Link structure: mostly same-host, some cross-host, some links to private pages, and some
        # tracking-parameter variants that should normalize to the same URL.
        self.links: dict[str, list[str]] = {}
        for url in urls:
            host = url.split("/")[2]
            out = []
            for _ in range(self.links_per_page):
                target_host = rng.choice(self.hosts) if rng.random() < self.cross_host_ratio else host
                target = f"http://{target_host}/page/{rng.randrange(self.pages_per_host)}"
                if rng.random() < 0.15:
                    target += f"?utm_source=newsletter{rng.randint(1, 9)}"
                out.append(target)
            if rng.random() < 0.3:
                out.append(f"http://{host}/private/{rng.randrange(self.private_pages)}")
            self.links[url] = out
        self.requests: list[tuple[str, float]] = []

    @property
    def public_pages(self) -> int:
        return len(self.pages)

    def cluster(self, url: str) -> str:
        """Duplicate-cluster id: an original and all of its copies share one."""
        return self.source.get(url, url)

    def _html(self, url: str) -> str:
        anchors = "".join(f'<a href="{t}">link</a> ' for t in self.links[url])
        body = f"<body>{_CHROME}<p>{self.pages[url]}</p><nav>{anchors}</nav>{_FOOTER}</body>"
        return f"<html><head><title>{url}</title></head>{body}</html>"

    async def handler(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.latency)
        host, path = request.url.host, request.url.path
        if path == "/robots.txt":
            delay = self.crawl_delay.get(host)
            body = "User-agent: *\nDisallow: /private/\n" + (f"Crawl-delay: {delay}\n" if delay else "")
            return httpx.Response(200, text=body, headers={"content-type": "text/plain"})
        url = f"http://{host}{path}"
        if url in self.private:
            return httpx.Response(200, html="<html><body>secret</body></html>")
        if url not in self.pages:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, html=self._html(url))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
