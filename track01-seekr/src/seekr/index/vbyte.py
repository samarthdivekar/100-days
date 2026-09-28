"""Variable-byte (VByte) integer coding: pure Python for short lists, NumPy for long ones.

Each integer is written 7 bits per byte, least-significant group first; the high bit of a byte is
1 when more bytes follow and 0 on the last byte. Small numbers take one byte, which is why
postings store *gaps* between sorted doc ids (and between positions) instead of the ids.

Why two implementations: term frequencies follow Zipf's law, so most postings lists hold one or
two documents. A NumPy call costs ~100 us of fixed overhead even for one integer, while a Python
loop costs ~0.7 us; a first version that always used NumPy spent minutes per 4M-token block on
per-call overhead. Measured crossovers (docs/day02-index.md): encode ~1,000 values, decode ~500 bytes.
"""

from __future__ import annotations

import numpy as np

_MAX_BYTES = 10  # enough for any uint64
ENCODE_NUMPY_MIN = 1024  # values
DECODE_NUMPY_MIN = 512  # bytes


def _encode_py(values) -> bytes:
    out = bytearray()
    for x in values:
        x = int(x)
        while x >= 128:
            out.append((x & 127) | 128)
            x >>= 7
        out.append(x)
    return bytes(out)


def _encode_np(v: np.ndarray) -> bytes:
    nbytes = np.ones(v.size, dtype=np.int64)
    for k in range(1, _MAX_BYTES):
        nbytes += v >= np.uint64(1 << (7 * k))
    ends = np.cumsum(nbytes)
    starts = ends - nbytes
    out = np.zeros(int(ends[-1]), dtype=np.uint8)
    for k in range(int(nbytes.max())):
        has = nbytes > k
        group = ((v[has] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        more = (nbytes[has] > k + 1).astype(np.uint8) << 7
        out[starts[has] + k] = group | more
    return out.tobytes()


def encode(values) -> bytes:
    """Encode a sequence of non-negative integers (list or array)."""
    n = len(values)
    if n == 0:
        return b""
    if n < ENCODE_NUMPY_MIN:
        return _encode_py(values.tolist() if isinstance(values, np.ndarray) else values)
    return _encode_np(np.asarray(values, dtype=np.uint64))


def _decode_py(buf) -> list[int]:
    out = []
    x = shift = 0
    for byte in bytes(buf):
        x |= (byte & 127) << shift
        if byte < 128:
            out.append(x)
            x = shift = 0
        else:
            shift += 7
    return out


def _decode_np(b: np.ndarray) -> np.ndarray:
    """Loop over byte *length* (at most a few passes) instead of doing per-byte index arithmetic.

    Nearly every gap fits in one byte, so the fast path returns the bytes as-is; otherwise each
    pass folds in one more 7-bit group for the values that have it. 3-24x faster than the first
    version on real postings (docs/day02-index.md).
    """
    last = b < 128  # final byte of each integer
    if last.all():
        return b.astype(np.uint64)
    ends = np.flatnonzero(last)
    lengths = np.diff(ends, prepend=-1)
    values = (b[ends] & 0x7F).astype(np.uint64)  # the final byte holds the most significant group
    for j in range(1, int(lengths.max())):
        longer = lengths > j
        values[longer] = (values[longer] << np.uint64(7)) | (b[ends[longer] - j] & 0x7F).astype(np.uint64)
    return values


def decode(buf) -> np.ndarray:
    size = buf.size if isinstance(buf, np.ndarray) else len(buf)
    if size == 0:
        return np.zeros(0, dtype=np.uint64)
    if size < DECODE_NUMPY_MIN:
        return np.array(_decode_py(buf), dtype=np.uint64)
    b = buf if isinstance(buf, np.ndarray) else np.frombuffer(buf, dtype=np.uint8)
    return _decode_np(b)


def gaps(sorted_values) -> list[int]:
    prev = 0
    out = []
    for x in sorted_values:
        out.append(x - prev)
        prev = x
    return out


def encode_gaps(sorted_values) -> bytes:
    n = len(sorted_values)
    if n == 0:
        return b""
    if n < ENCODE_NUMPY_MIN:
        vals = sorted_values.tolist() if isinstance(sorted_values, np.ndarray) else sorted_values
        return _encode_py(gaps(vals))
    v = np.asarray(sorted_values, dtype=np.uint64)
    return _encode_np(np.diff(v, prepend=np.uint64(0)))


def decode_gaps(buf) -> np.ndarray:
    return np.cumsum(decode(buf), dtype=np.uint64)
