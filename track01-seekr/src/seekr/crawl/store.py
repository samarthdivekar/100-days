"""SQLite storage for crawled pages and the link graph.

Day 2 indexes `pages`; Day 4 runs PageRank over `links`.
"""

from __future__ import annotations

import json
import sqlite3
import time
import zlib
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS pages (
    url          TEXT PRIMARY KEY,   -- normalized URL we requested
    final_url    TEXT,               -- after redirects
    status       INTEGER,
    depth        INTEGER,
    fetched_at   REAL,
    title        TEXT,
    text         TEXT,
    content_hash TEXT,
    simhash      TEXT,               -- 64-bit fingerprint as hex (SQLite INTEGER is signed)
    dup_of       TEXT,               -- url of the page this duplicates, if any
    dup_kind     TEXT,               -- 'exact' | 'near' | NULL
    noindex      INTEGER DEFAULT 0,
    outlinks     INTEGER DEFAULT 0,
    html         BLOB                -- zlib-compressed raw HTML: reprocess without refetching
);
CREATE INDEX IF NOT EXISTS idx_pages_hash ON pages(content_hash);
CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, dst)
);
CREATE TABLE IF NOT EXISTS crawl_runs (
    id         INTEGER PRIMARY KEY,
    started_at REAL,
    stats      TEXT
);
"""


class PageStore:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(pages)")}
        if "html" not in cols:
            self.conn.execute("ALTER TABLE pages ADD COLUMN html BLOB")

    def save_page(
        self, *, url, final_url, status, depth, page, content_hash, simhash, dup_of, dup_kind, html=None
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO pages (url, final_url, status, depth, fetched_at, title, text,
                                             content_hash, simhash, dup_of, dup_kind, noindex, outlinks, html)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                url,
                final_url,
                status,
                depth,
                time.time(),
                page.title if page else "",
                page.text if page else "",
                content_hash,
                f"{simhash:016x}" if simhash is not None else None,
                dup_of,
                dup_kind,
                int(bool(page and page.noindex)),
                len(page.links) if page else 0,
                zlib.compress(html.encode("utf-8"), 6) if html else None,
            ),
        )
        if page and page.links:
            self.conn.executemany(
                "INSERT OR IGNORE INTO links (src, dst) VALUES (?, ?)", [(url, d) for d in page.links]
            )

    def html(self, url: str) -> str | None:
        row = self.conn.execute("SELECT html FROM pages WHERE url = ?", (url,)).fetchone()
        return zlib.decompress(row[0]).decode("utf-8") if row and row[0] else None

    def save_run(self, stats: dict) -> None:
        self.conn.execute(
            "INSERT INTO crawl_runs (started_at, stats) VALUES (?, ?)", (time.time(), json.dumps(stats))
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
