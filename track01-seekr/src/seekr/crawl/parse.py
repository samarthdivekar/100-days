"""HTML -> title, text, main-content text, outgoing links, and the page's own robots directives.

Two text views are extracted:

* `text`: every visible word (what Day 2 indexes).
* `main_text`: the same minus page chrome: <nav>, <header>, <footer>, <aside>, and elements
  whose class or id says navigation, menu, sidebar, breadcrumb, banner, cookie... Duplicate
  detection fingerprints this. On a real templated site (books.toscrape.com) fingerprinting the
  full text flagged 27 of 200 pages as near-duplicates, mostly category pages listing
  completely different books that only shared a 50-link sidebar (see docs/day01-crawler.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from seekr.crawl.urls import normalize

_SKIP = {"script", "style", "noscript", "template", "svg"}
_BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "td"}
_CHROME_TAGS = {"nav", "header", "footer", "aside"}
_CHROME_ATTR = re.compile(
    r"(^|[\s_-])(nav|navbar|menu|sidebar|side|breadcrumbs?|footer|header|banner|cookie|masthead|toolbar)"
    r"([\s_-]|$)",
    re.I,
)
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


@dataclass
class Page:
    title: str = ""
    text: str = ""
    main_text: str = ""
    links: list[str] = field(default_factory=list)  # normalized, followable
    canonical: str | None = None
    noindex: bool = False
    nofollow: bool = False


def _clean(chunks: list[str]) -> str:
    lines = (" ".join(line.split()) for line in "".join(chunks).splitlines())
    return "\n".join(line for line in lines if line)


class _Extractor(HTMLParser):
    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.page = Page()
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, is_skip, is_chrome)
        self._skip = 0
        self._chrome = 0
        self._in_title = False
        self._all: list[str] = []
        self._main: list[str] = []
        self._title: list[str] = []

    def _emit(self, s: str) -> None:
        self._all.append(s)
        if not self._chrome:
            self._main.append(s)

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "base" and a.get("href"):
            self.base = normalize(a["href"], self.base) or self.base
        elif tag == "meta" and a.get("name", "").lower() in ("robots", "seekrbot"):
            content = a.get("content", "").lower()
            self.page.noindex |= "noindex" in content or "none" in content
            self.page.nofollow |= "nofollow" in content or "none" in content
        elif tag == "link" and "canonical" in a.get("rel", "").lower().split():
            self.page.canonical = normalize(a.get("href", ""), self.base)
        elif tag == "a" and a.get("href") and "nofollow" not in a.get("rel", "").lower().split():
            url = normalize(a["href"], self.base)
            if url:
                self.page.links.append(url)
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK:
            self._emit("\n")
        if tag in _VOID:
            return
        is_skip = tag in _SKIP
        is_chrome = tag in _CHROME_TAGS or bool(
            _CHROME_ATTR.search(a.get("class", "")) or _CHROME_ATTR.search(a.get("id", ""))
        )
        self._stack.append((tag, is_skip, is_chrome))
        self._skip += is_skip
        self._chrome += is_chrome

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK:
            self._emit("\n")
        if tag in _VOID or not any(t == tag for t, _, _ in self._stack):
            return
        # Pop until the matching open tag: HTML in the wild leaves elements unclosed.
        while self._stack:
            t, is_skip, is_chrome = self._stack.pop()
            self._skip -= is_skip
            self._chrome -= is_chrome
            if t == tag:
                break

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        elif not self._skip:
            self._emit(data)


def parse_html(html: str, base_url: str) -> Page:
    ex = _Extractor(base_url)
    try:
        ex.feed(html)
        ex.close()
    except Exception:  # malformed HTML: keep whatever was extracted
        pass
    page = ex.page
    page.title = " ".join("".join(ex._title).split())
    page.text = _clean(ex._all)
    page.main_text = _clean(ex._main)
    page.links = list(dict.fromkeys(page.links))
    return page


def analyze(html: str, url: str):
    """All per-page CPU work in one picklable call, so it can run in a worker process.

    Returns (page, content hash, SimHash, MinHash signature). Duplicates are judged on main
    content; a page that is all chrome falls back to its full text.
    """
    from seekr.crawl.dedup import content_hash, fingerprint

    page = parse_html(html, url)
    basis = page.main_text or page.text
    fp, sig = fingerprint(basis)
    return page, content_hash(basis), fp, sig
