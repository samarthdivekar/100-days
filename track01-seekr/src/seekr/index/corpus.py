"""Document sources: the Day 1 crawl database and Wikimedia CirrusSearch dumps."""

from __future__ import annotations

import gzip
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from seekr.index.builder import Document


def from_crawl(db_path: Path | str) -> Iterator[Document]:
    """Crawled pages, minus duplicates and pages that asked not to be indexed."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT url, title, text FROM pages
           WHERE status BETWEEN 200 AND 299 AND dup_of IS NULL AND noindex = 0 ORDER BY fetched_at"""
    )
    for url, title, text in rows:
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
