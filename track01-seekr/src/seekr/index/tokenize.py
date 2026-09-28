"""Text -> terms. Unicode-normalized, case-folded, positions kept (for phrase queries).

No stemming or stopword removal yet: positions make "to be or not to be" answerable, and Day 3
measures whether stemming actually helps ranking before adding it.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)  # letters and digits; underscores and punctuation split
MAX_TERM_LEN = 40


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return [t for t in _TOKEN_RE.findall(text) if len(t) <= MAX_TERM_LEN]
