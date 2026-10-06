"""Document sources: the Day 1 crawl database and Wikimedia CirrusSearch dumps."""

from __future__ import annotations

import gzip
import json
import sqlite3
import zlib
from collections.abc import Iterator
from pathlib import Path

from seekr.index.builder import Document


def from_crawl(db_path: Path | str, main_content: bool = True) -> Iterator[Document]:
    """Crawled pages, minus duplicates and pages that asked not to be indexed.

    With `main_content` (the default), each page is re-parsed from its stored HTML and only the
    main content is indexed: menus, sidebars and footers repeat on every page of a site, and on
    books.toscrape.com the 50-category sidebar made "poetry" match all 38 pages (docs/day03-bm25.md).
    """
    from seekr.crawl.parse import parse_html

    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(pages)")}
    html_col = "html" if "html" in cols else "NULL"
    rows = conn.execute(
        f"""SELECT url, final_url, title, text, {html_col} FROM pages
            WHERE status BETWEEN 200 AND 299 AND dup_of IS NULL AND noindex = 0 ORDER BY fetched_at"""
    )
    for url, final_url, title, text, html in rows:
        if main_content and html:
            page = parse_html(zlib.decompress(html).decode("utf-8"), final_url or url)
            text = page.main_text or page.text
        yield Document(url=url, title=title or "", text=text or "")


def from_cirrus(path: Path | str, limit: int | None = None) -> Iterator[Document]:
    """Wikimedia CirrusSearch content dump: alternating action and document JSON lines.

    Articles come with plain text already extracted (no wikitext to clean), plus metadata such as
    `incoming_links` that Day 4 compares against PageRank.
    """
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith('{"index"'):
                continue
            doc = json.loads(line)
            if doc.get("namespace", 0) != 0 or not doc.get("text"):
                continue
            title = doc.get("title", "")
            url = "https://simple.wikipedia.org/wiki/" + title.replace(" ", "_")
            yield Document(url=url, title=title, text=doc["text"])
            n += 1
            if limit and n >= limit:
                return
