"""Read side: memory-mapped postings + SQLite lexicon and doc table."""

from __future__ import annotations

import json
import mmap
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seekr.index import vbyte


@dataclass(frozen=True)
class TermInfo:
    term: str
    df: int
    cf: int
    offset: int
    docs_len: int
    tfs_len: int
    pos_len: int


class IndexReader:
    def __init__(self, index_dir: Path | str):
        self.dir = Path(index_dir)
        self.db = sqlite3.connect(self.dir / "index.db", check_same_thread=False)
        self._file = open(self.dir / "postings.bin", "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ) if self._size() else None
        self.meta = {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}
        self.n_docs = int(self.meta["docs"])
        self._doc_lengths: np.ndarray | None = None
        self.skip_every = int(self.meta.get("skip_every", 0))
        skips_path = self.dir / "skips.bin"
        self._skips = (
            np.fromfile(skips_path, dtype=np.uint32) if self.skip_every and skips_path.exists() else None
        )

    def _size(self) -> int:
        return (self.dir / "postings.bin").stat().st_size

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
        self._file.close()
        self.db.close()

    def term(self, term: str) -> TermInfo | None:
        row = self.db.execute("SELECT * FROM terms WHERE term = ?", (term,)).fetchone()
        return TermInfo(*row) if row else None

    def _slice(self, start: int, length: int) -> np.ndarray:
        return np.frombuffer(self._mm, dtype=np.uint8, count=length, offset=start)

    def docs(self, info: TermInfo) -> np.ndarray:
        return vbyte.decode_gaps(self._slice(info.offset, info.docs_len)).astype(np.int64)

    def tfs(self, info: TermInfo) -> np.ndarray:
        return vbyte.decode(self._slice(info.offset + info.docs_len, info.tfs_len)).astype(np.int64)

    def skip_offsets(self, term: str) -> np.ndarray | None:
        if self._skips is None:
            return None
        row = self.db.execute("SELECT offset, count FROM skips WHERE term = ?", (term,)).fetchone()
        return None if row is None else self._skips[row[0] // 4 : row[0] // 4 + row[1]]

    def position_keys(self, info: TermInfo, candidates: np.ndarray | None = None) -> np.ndarray:
        """Occurrences of the term as int64 keys doc_id << 32 | position (sorted).

        With `candidates` (sorted doc ids, all containing the term) and a skip table, only the
        128-document runs holding a candidate are decoded; keys for other documents in those runs
        may be included, so callers still filter by document.
        """
        docs = self.docs(info)
        tfs = self.tfs(info)
        pos = self._slice(info.offset + info.docs_len + info.tfs_len, info.pos_len)
        skips = self.skip_offsets(info.term) if candidates is not None else None
        if skips is not None:
            chunks = np.unique(np.searchsorted(docs, candidates) // self.skip_every)
            if chunks.size < 0.5 * skips.size:
                stops = np.append(skips[1:], np.uint32(info.pos_len))
                pos = np.concatenate([pos[skips[c] : stops[c]] for c in chunks])
                keep = np.concatenate(
                    [
                        np.arange(c * self.skip_every, min((c + 1) * self.skip_every, docs.size))
                        for c in chunks
                    ]
                )
                docs, tfs = docs[keep], tfs[keep]
        gaps = vbyte.decode(pos).astype(np.int64)
        # undo per-document delta coding: cumulative sum that restarts at each document
        starts = np.concatenate(([0], np.cumsum(tfs)[:-1]))
        csum = np.cumsum(gaps)
        base = np.repeat(csum[starts] - gaps[starts], tfs)
        positions = csum - base
        return (np.repeat(docs, tfs) << 32) | positions

    def doc_lengths(self) -> np.ndarray:
        if self._doc_lengths is None:
            lengths = np.zeros(self.n_docs, dtype=np.int64)
            for doc_id, length in self.db.execute("SELECT id, length FROM docs"):
                lengths[doc_id] = length
            self._doc_lengths = lengths
        return self._doc_lengths

    def doc(self, doc_id: int) -> dict:
        row = self.db.execute("SELECT url, title, length, text FROM docs WHERE id = ?", (doc_id,)).fetchone()
        url, title, length, blob = row
        return {
            "id": doc_id,
            "url": url,
            "title": title,
            "length": length,
            "text": zlib.decompress(blob).decode("utf-8") if blob else None,
        }
