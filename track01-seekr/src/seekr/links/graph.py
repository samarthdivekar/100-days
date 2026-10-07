"""Build the article link graph from a Wikimedia CirrusSearch dump, aligned with the index's doc ids.

Two streaming passes over the dump, so the graph never needs every link string in memory:

1. Assign doc ids exactly as `corpus.from_cirrus` does (namespace 0, non-empty text, dump order),
   and map every title *and redirect alias* to its doc id. Wikipedia links often point at a
   redirect ("USA" -> "United States"); without the alias map those links would be dropped.
2. Resolve each article's `outgoing_link` titles to doc ids. Links to pages outside the corpus
   (red links, other namespaces) are dropped; repeated links and self-links count once / not at all.

Also kept, per doc id: Wikipedia's own `incoming_links` count and `popularity_score` (a pageview
share), the two external signals Day 4 validates PageRank against.
"""

from __future__ import annotations

import gzip
import json
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalize_title(title: str) -> str:
    """MediaWiki title canonical form: spaces not underscores, first letter uppercase."""
    t = title.replace("_", " ").strip()
    return t[:1].upper() + t[1:] if t else t


@dataclass
class LinkGraph:
    src: np.ndarray  # int32, edge sources (doc ids)
    dst: np.ndarray  # int32, edge targets
    n: int
    titles: list[str]
    incoming_links: np.ndarray  # Wikipedia's count, for validation
    popularity: np.ndarray  # pageview share, for validation
    stats: dict

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, src=self.src, dst=self.dst, incoming_links=self.incoming_links, popularity=self.popularity
        )

    @classmethod
    def load(cls, path: Path, titles: list[str] | None = None) -> LinkGraph:
        z = np.load(path)
        n = len(z["incoming_links"])
        return cls(z["src"], z["dst"], n, titles or [], z["incoming_links"], z["popularity"], {})


def _articles(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith('{"index"'):
                continue
            doc = json.loads(line)
            if doc.get("namespace", 0) != 0 or not doc.get("text"):
                continue  # same filter as corpus.from_cirrus, so doc ids line up with the index
            yield doc


def build_link_graph(dump: Path | str, limit: int | None = None) -> LinkGraph:
    dump = Path(dump)
    titles: list[str] = []
    by_title: dict[str, int] = {}
    incoming, popularity = array("q"), array("d")  # "q": int64 on every OS ("l" is 32-bit on Windows)
    aliases = 0
    for doc_id, doc in enumerate(_articles(dump)):
        if limit and doc_id >= limit:
            break
        title = normalize_title(doc.get("title", ""))
        titles.append(title)
        by_title[title] = doc_id
        for r in doc.get("redirect") or []:
            if r.get("namespace", 0) == 0:
                by_title.setdefault(normalize_title(r["title"]), doc_id)
                aliases += 1
        incoming.append(int(doc.get("incoming_links") or 0))
        popularity.append(float(doc.get("popularity_score") or 0.0))

    src, dst = array("i"), array("i")
    raw = unresolved = self_links = 0
    for doc_id, doc in enumerate(_articles(dump)):
        if limit and doc_id >= limit:
            break
        seen: set[int] = set()
        for target in doc.get("outgoing_link") or []:
            raw += 1
            t = by_title.get(normalize_title(target))
            if t is None:
                unresolved += 1
            elif t == doc_id:
                self_links += 1
            elif t not in seen:
                seen.add(t)
                src.append(doc_id)
                dst.append(t)
    n = len(titles)
    stats = {
        "articles": n,
        "redirect_aliases": aliases,
        "links_in_dump": raw,
        "edges": len(src),
        "unresolved_links": unresolved,
        "self_links": self_links,
    }
    return LinkGraph(
        np.frombuffer(src, dtype=np.int32).copy(),
        np.frombuffer(dst, dtype=np.int32).copy(),
        n,
        titles,
        np.frombuffer(incoming, dtype=np.int64).copy(),
        np.frombuffer(popularity, dtype=np.float64).copy(),
        stats,
    )
