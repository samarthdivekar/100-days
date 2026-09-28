"""Skip tables for positions: decode only the stretches of a long positions list that a query needs.

A phrase like "the city" used to decode all 3.4M positions of "the" to check the 37k documents
that also contain "city". For every term with df >= `min_df`, we record where each run of
`every` documents starts inside its positions stream. Because each document's positions begin
with an absolute value, those runs can be sliced out, concatenated, and decoded in one call.

Skip tables are derived from postings.bin alone (a document's position run ends at the byte that
closes its tf-th integer), so an existing index can gain them without a rebuild.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from seekr.index import vbyte

SKIP_EVERY = 128
SKIP_MIN_DF = 1024


def build_skips(index_dir: Path | str, min_df: int = SKIP_MIN_DF, every: int = SKIP_EVERY) -> dict:
    index_dir = Path(index_dir)
    db = sqlite3.connect(index_dir / "index.db")
    db.execute("DROP TABLE IF EXISTS skips")
    db.execute("CREATE TABLE skips (term TEXT PRIMARY KEY, offset INTEGER, count INTEGER) WITHOUT ROWID")
    rows = db.execute(
        "SELECT term, offset, docs_len, tfs_len, pos_len FROM terms WHERE df >= ?", (min_df,)
    ).fetchall()
    entries = []
    written = 0
    postings = np.memmap(index_dir / "postings.bin", dtype=np.uint8, mode="r") if rows else None
    with open(index_dir / "skips.bin", "wb") as out:
        for term, offset, docs_len, tfs_len, pos_len in rows:
            start = offset + docs_len
            tfs = vbyte.decode(postings[start : start + tfs_len]).astype(np.int64)
            pos = postings[start + tfs_len : start + tfs_len + pos_len]
            ends = np.flatnonzero(pos < 128)  # last byte of every position integer
            cum = np.cumsum(tfs)
            doc_start = np.concatenate(([0], ends[cum[:-1] - 1] + 1))  # byte where each doc's run begins
            skips = doc_start[::every].astype(np.uint32)
            out.write(skips.tobytes())
            entries.append((term, written, skips.size))
            written += skips.nbytes
    del postings
    db.executemany("INSERT INTO skips VALUES (?, ?, ?)", entries)
    db.execute("INSERT OR REPLACE INTO meta VALUES ('skip_every', ?)", (str(every),))
    db.commit()
    db.close()
    return {"terms_with_skips": len(entries), "skip_bytes": written}
