"""URL normalization: many spellings of one page must map to one key, or the crawler refetches it.

HTTP://Example.com:80/a/./b/../c?utm_source=x&b=2&a=1#frag  ->  http://example.com/a/c?a=1&b=2
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
    }
)
DEFAULT_PORTS = {"http": 80, "https": 443}
# Characters left unescaped in a path (RFC 3986 unreserved + sub-delims + ':' '@' '/').
_PATH_SAFE = "/:@!$&'()*+,;=-._~"


def _remove_dot_segments(path: str) -> str:
    out: list[str] = []
    for seg in path.split("/"):
        if seg == "..":
            if len(out) > 1:
                out.pop()
        elif seg != ".":
            out.append(seg)
    result = "/".join(out)
    if path.endswith(("/.", "/..")):
        result += "/"
    return result if result.startswith("/") else "/" + result


def normalize(url: str, base: str | None = None) -> str | None:
    """Canonical form of an http(s) URL, or None if it isn't crawlable."""
    url = url.strip()
    if base:
        url = urljoin(base, url)
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    if scheme not in DEFAULT_PORTS or not host:
        return None
    netloc = host if port in (None, DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    # Decode then re-encode so %7E and ~ (or %2f vs %2F) compare equal; keep an encoded slash encoded.
    raw_path = parts.path.replace("%2F", "\x00").replace("%2f", "\x00")
    path = quote(unquote(raw_path), safe=_PATH_SAFE).replace("%00", "%2F")
    path = _remove_dot_segments(re.sub(r"/{2,}", "/", path or "/"))
    query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS
    ]
    return urlunsplit((scheme, netloc, path, urlencode(sorted(query)), ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc


def path_and_query(url: str) -> str:
    parts = urlsplit(url)
    return (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
