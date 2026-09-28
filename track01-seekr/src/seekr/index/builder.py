"""Memory-bounded index construction (SPIMI: single-pass in-memory indexing, Heinz & Zobel 2003).

    documents ─▶ batches of ~`block_tokens` tokens ─▶ (worker processes) sorted, compressed blocks
                                                                         │
    all blocks ─▶ k-way merge by term ─▶ postings.bin + lexicon + doc table

Postings for one term, stored contiguously in postings.bin:

    docs:      VByte gaps between sorted doc ids
    tfs:       VByte term frequency per doc
    positions: per doc, the first position absolute and the rest as gaps, all docs concatenated

Because every doc's position run starts absolute and tf values are plain VByte, the positions
and tf streams of two blocks can be merged by concatenating bytes; only the doc-id stream needs
re-gapping where one block ends and the next begins.
"""

from __future__ import annotations

import heapq
import json
import shutil
import sqlite3
import struct
import time
import zlib
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seekr.index import vbyte
from seekr.index.skips import build_skips
from seekr.index.tokenize import tokenize

_ENTRY = struct.Struct("<HIQIII")  # term_len, df, cf, docs_len, tfs_len, pos_len


@dataclass
class Document:
    url: str
    title: str
    text: str


def _positions_stream(pos_lists: list[list[int]]) -> bytes:
    """Per document: first position absolute, the rest as gaps; all documents concatenated."""
    total = sum(len(ps) for ps in pos_lists)
    if total < vbyte.ENCODE_NUMPY_MIN:  # the common case: rare terms, short lists
        out: list[int] = []
        for ps in pos_lists:
            out.extend(vbyte.gaps(ps))
        return vbyte.encode(out)
    flat = np.fromiter((p for ps in pos_lists for p in ps), dtype=np.uint64, count=total)
    lengths = np.fromiter((len(ps) for ps in pos_lists), dtype=np.int64, count=len(pos_lists))
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    gaps = np.diff(flat, prepend=np.uint64(0))
    gaps[starts] = flat[starts]  # each document's first position is absolute
    return vbyte.encode(gaps)


def _write_block(path: Path, postings: dict[str, tuple[list, list, list]]) -> None:
    with open(path, "wb") as f:
        for term in sorted(postings):
            docs, tfs, pos = postings[term]
            t = term.encode("utf-8")
            d = vbyte.encode_gaps(docs)
            tf = vbyte.encode(tfs)
            p = _positions_stream(pos)
            f.write(_ENTRY.pack(len(t), len(docs), sum(tfs), len(d), len(tf), len(p)))
            f.write(t)
            f.write(d)
            f.write(tf)
            f.write(p)


def _read_block(path: Path) -> Iterator[tuple[str, int, int, bytes, bytes, bytes]]:
    with open(path, "rb") as f:
        while header := f.read(_ENTRY.size):
            tlen, df, cf, dlen, tflen, plen = _ENTRY.unpack(header)
            term = f.read(tlen).decode("utf-8")
            yield term, df, cf, f.read(dlen), f.read(tflen), f.read(plen)


def _build_block(path: str, first_id: int, docs: list[tuple[str, str]], store_text: bool):
    """Index one batch of documents into a sorted, compressed block file (runs in a worker process).

    Returns per-document token counts, compressed texts, and the batch's raw byte count.
    """
    postings: dict[str, tuple[list, list, list]] = {}
    lengths: list[int] = []
    blobs: list[bytes | None] = []
    raw = 0
    for offset, (title, text) in enumerate(docs):
        doc_id = first_id + offset
        # Title words are indexed before the body so a title phrase is searchable too.
        tokens = tokenize(title + "\n" + text)
        positions: dict[str, list[int]] = {}
        for i, tok in enumerate(tokens):
            positions.setdefault(tok, []).append(i)
        for term, ps in positions.items():
            entry = postings.get(term)
            if entry is None:
                entry = postings[term] = ([], [], [])
            entry[0].append(doc_id)
            entry[1].append(len(ps))
            entry[2].append(ps)
        encoded = text.encode("utf-8")
        raw += len(encoded)
        lengths.append(len(tokens))
        blobs.append(zlib.compress(encoded, 6) if store_text else None)
    _write_block(Path(path), postings)
    return lengths, blobs, raw


class IndexBuilder:
    """Batches documents into blocks; blocks are built inline or by a process pool, then merged.

    Block boundaries depend only on the input (a batch closes once its text is estimated to hold
    `block_tokens` tokens), and blocks are merged in batch order, so the index is byte-identical
    whatever `workers` is. A test enforces that.
    """

    CHARS_PER_TOKEN = 6  # estimate used to size batches before tokenizing

    def __init__(
        self,
        out_dir: Path | str,
        block_tokens: int = 2_000_000,
        store_text: bool = True,
        workers: int = 0,
    ):
        self.out = Path(out_dir)
        if self.out.exists():
            shutil.rmtree(self.out)
        self.tmp = self.out / "blocks"
        self.tmp.mkdir(parents=True)
        self.block_tokens = block_tokens
        self.store_text = store_text
        self.workers = workers
        self.db = sqlite3.connect(self.out / "index.db")
        self.db.executescript(
            """CREATE TABLE docs (id INTEGER PRIMARY KEY, url TEXT, title TEXT, length INTEGER, text BLOB);
               CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"""
        )
        self._batch: list[Document] = []
        self._batch_chars = 0
        self._blocks: list[Path] = []
        self._pending: deque = deque()  # (path, first_id, docs, future) in submission order
        self._pool = ProcessPoolExecutor(max_workers=workers) if workers > 0 else None
        self.n_assigned = 0
        self.n_docs = 0
        self.n_tokens = 0
        self.raw_bytes = 0
        self.t0 = time.perf_counter()

    def add(self, doc: Document) -> int:
        doc_id = self.n_assigned
        self.n_assigned += 1
        self._batch.append(doc)
        self._batch_chars += len(doc.title) + len(doc.text) + 1
        if self._batch_chars >= self.block_tokens * self.CHARS_PER_TOKEN:
            self._submit()
        return doc_id

    def add_all(self, docs: Iterable[Document], progress_every: int = 0) -> None:
        for i, doc in enumerate(docs, 1):
            self.add(doc)
            if progress_every and i % progress_every == 0:
                rate = i / (time.perf_counter() - self.t0)
                print(f"  {i:>9,} docs read  {self.n_tokens:>12,} tokens indexed  {rate:,.0f} docs/s")

    def _submit(self) -> None:
        if not self._batch:
            return
        docs, self._batch, self._batch_chars = self._batch, [], 0
        first_id = self.n_assigned - len(docs)
        path = self.tmp / f"block-{len(self._blocks) + len(self._pending):05d}.bin"
        payload = [(d.title, d.text) for d in docs]
        if self._pool is None:
            self._record(path, first_id, docs, _build_block(str(path), first_id, payload, self.store_text))
            return
        future = self._pool.submit(_build_block, str(path), first_id, payload, self.store_text)
        self._pending.append((path, first_id, docs, future))
        while len(self._pending) > 2 * self.workers:  # bound memory: at most 2 batches per worker in flight
            self._complete_oldest()

    def _complete_oldest(self) -> None:
        path, first_id, docs, future = self._pending.popleft()
        self._record(path, first_id, docs, future.result())

    def _record(self, path: Path, first_id: int, docs: list[Document], result) -> None:
        lengths, blobs, raw = result
        rows = zip(docs, lengths, blobs, strict=True)
        self.db.executemany(
            "INSERT INTO docs VALUES (?, ?, ?, ?, ?)",
            [(first_id + i, d.url, d.title, n, b) for i, (d, n, b) in enumerate(rows)],
        )
        self._blocks.append(path)
        self.n_docs += len(docs)
        self.n_tokens += sum(lengths)
        self.raw_bytes += raw

    def finish(self) -> dict:
        self._submit()
        while self._pending:
            self._complete_oldest()
        if self._pool is not None:
            self._pool.shutdown()
        t_merge = time.perf_counter()
        lexicon = []
        offset = 0
        streams = [_read_block(p) for p in self._blocks]
        with open(self.out / "postings.bin", "wb") as out:
            current = None
            parts: list[tuple] = []

            def emit():
                nonlocal offset
                term = current
                if len(parts) == 1:
                    _, df, cf, d, tf, p = parts[0]
                else:  # re-gap doc ids across block boundaries; tf and position bytes concatenate
                    ids = np.concatenate([vbyte.decode_gaps(x[3]) for x in parts])
                    d = vbyte.encode_gaps(ids)
                    df = sum(x[1] for x in parts)
                    cf = sum(x[2] for x in parts)
                    tf = b"".join(x[4] for x in parts)
                    p = b"".join(x[5] for x in parts)
                out.write(d)
                out.write(tf)
                out.write(p)
                lexicon.append((term, df, cf, offset, len(d), len(tf), len(p)))
                offset += len(d) + len(tf) + len(p)

            for entry in heapq.merge(*streams, key=lambda e: e[0]):
                if entry[0] != current and current is not None:
                    emit()
                    parts = []
                current = entry[0]
                parts.append(entry)
            if current is not None:
                emit()

        self.db.execute(
            """CREATE TABLE terms (term TEXT PRIMARY KEY, df INTEGER, cf INTEGER, offset INTEGER,
                                   docs_len INTEGER, tfs_len INTEGER, pos_len INTEGER) WITHOUT ROWID"""
        )
        self.db.executemany("INSERT INTO terms VALUES (?, ?, ?, ?, ?, ?, ?)", lexicon)
        n_postings = sum(x[1] for x in lexicon)
        stats = {
            "docs": self.n_docs,
            "tokens": self.n_tokens,
            "terms": len(lexicon),
            "postings": n_postings,
            "blocks": len(self._blocks),
            "raw_text_bytes": self.raw_bytes,
            "postings_bytes": offset,
            "docid_bytes": sum(x[4] for x in lexicon),
            "tf_bytes": sum(x[5] for x in lexicon),
            "position_bytes": sum(x[6] for x in lexicon),
            # what the same data costs as fixed 32-bit integers: doc id + tf per posting, one per position
            "uncompressed_bytes": 4 * (2 * n_postings + self.n_tokens),
            "avg_doc_length": self.n_tokens / max(1, self.n_docs),
            "build_seconds": round(time.perf_counter() - self.t0, 1),
            "merge_seconds": round(time.perf_counter() - t_merge, 1),
        }
        self.db.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in stats.items()])
        self.db.commit()
        self.db.close()
        shutil.rmtree(self.tmp)
        stats.update(build_skips(self.out))
        return stats
