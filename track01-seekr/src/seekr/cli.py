"""`seekr <command>`: the command-line entry point. Each Seekr day adds a command."""

from __future__ import annotations

import argparse
import json
import sqlite3

from seekr.crawl.crawler import DEFAULT_AGENT, CrawlConfig, crawl
from seekr.crawl.store import PageStore
from seekr.crawl.urls import host_of, normalize


def cmd_crawl(args) -> None:
    seeds = [normalize(s) for s in args.seed]
    seeds = [s for s in seeds if s]
    allowed = None if args.anywhere else {host_of(s) for s in seeds}
    cfg = CrawlConfig(
        seeds=seeds,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        concurrency=args.concurrency,
        delay=args.delay,
        allowed_hosts=allowed,
        cpu_workers=args.cpu_workers,
        user_agent=DEFAULT_AGENT.replace(")", f"; {args.contact})") if args.contact else DEFAULT_AGENT,
    )
    store = PageStore(args.db)
    stats = crawl(cfg, store)
    store.close()
    print(json.dumps(stats.as_dict(), indent=2))


def cmd_stats(args) -> None:
    conn = sqlite3.connect(args.db)
    total, stored, exact, near, noindex = conn.execute(
        """SELECT COUNT(*), SUM(status BETWEEN 200 AND 299), SUM(dup_kind = 'exact'),
                  SUM(dup_kind = 'near'), SUM(noindex) FROM pages"""
    ).fetchone()
    links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    hosts = conn.execute(
        "SELECT COUNT(DISTINCT substr(url, 1, instr(substr(url, 9), '/') + 8)) FROM pages"
    ).fetchone()[0]
    print(f"pages: {total}  ok: {stored}  exact dups: {exact}  near dups: {near}  noindex: {noindex}")
    print(f"hosts: {hosts}  links: {links}")
    for row in conn.execute("SELECT stats FROM crawl_runs ORDER BY id DESC LIMIT 1"):
        print("last run:", row[0])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="seekr", description="A search engine built from scratch.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("crawl", help="crawl the web politely from seed URLs")
    c.add_argument("seed", nargs="+")
    c.add_argument("--db", default="data/crawl.db")
    c.add_argument("--max-pages", type=int, default=200)
    c.add_argument("--max-depth", type=int, default=5)
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--delay", type=float, default=1.0, help="minimum seconds between requests to one host")
    c.add_argument("--anywhere", action="store_true", help="follow links off the seed hosts")
    c.add_argument("--cpu-workers", type=int, default=0, help="parse/hash in N processes (large pages)")
    c.add_argument("--contact", help="URL or email added to the User-Agent so site owners can reach you")
    c.set_defaults(func=cmd_crawl)

    s = sub.add_parser("crawl-stats", help="summarize a crawl database")
    s.add_argument("--db", default="data/crawl.db")
    s.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
