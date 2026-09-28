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


def cmd_index(args) -> None:
    from seekr.index.builder import IndexBuilder
    from seekr.index.corpus import from_cirrus, from_crawl

    if bool(args.crawl_db) == bool(args.cirrus):
        raise SystemExit("give exactly one of --crawl-db or --cirrus")
    docs = from_crawl(args.crawl_db) if args.crawl_db else from_cirrus(args.cirrus, limit=args.limit)
    builder = IndexBuilder(args.out, block_tokens=args.block_tokens, workers=args.workers)
    builder.add_all(docs, progress_every=args.progress)
    print(json.dumps(builder.finish(), indent=2))


def cmd_search(args) -> None:
    import time

    from seekr.index.query import Searcher
    from seekr.index.reader import IndexReader

    reader = IndexReader(args.index)
    t0 = time.perf_counter()
    hits = Searcher(reader).search(args.query)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{len(hits):,} matching documents in {ms:.1f} ms")
    for doc_id in hits[: args.limit]:
        d = reader.doc(int(doc_id))
        print(f"  [{doc_id}] {d['title']}  {d['url']}")


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

    ix = sub.add_parser("index", help="build a compressed positional inverted index")
    ix.add_argument("--crawl-db", help="index a Day 1 crawl database")
    ix.add_argument("--cirrus", help="index a Wikimedia CirrusSearch content dump (.json.gz)")
    ix.add_argument("--out", default="data/index")
    ix.add_argument("--limit", type=int, help="stop after N documents")
    ix.add_argument("--block-tokens", type=int, default=2_000_000, help="SPIMI batch size (tokens)")
    ix.add_argument("--workers", type=int, default=0, help="build blocks in N processes (0 = inline)")
    ix.add_argument("--progress", type=int, default=20_000)
    ix.set_defaults(func=cmd_index)

    q = sub.add_parser("search", help='boolean/phrase search: raptor hedges, "new york" OR boston, -stock')
    q.add_argument("query")
    q.add_argument("--index", default="data/index")
    q.add_argument("--limit", type=int, default=10)
    q.set_defaults(func=cmd_search)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
