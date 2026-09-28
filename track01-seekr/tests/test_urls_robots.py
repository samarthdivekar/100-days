import pytest

from seekr.crawl.robots import RobotsPolicy, parse
from seekr.crawl.urls import normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("HTTP://Example.COM:80/a/./b/../c", "http://example.com/a/c"),
        ("https://example.com:443", "https://example.com/"),
        ("http://example.com:8080/x", "http://example.com:8080/x"),
        ("http://example.com/x#section", "http://example.com/x"),
        ("http://example.com/x?b=2&a=1&utm_source=mail", "http://example.com/x?a=1&b=2"),
        ("http://example.com/%7Euser", "http://example.com/~user"),
        ("http://example.com//double//slash", "http://example.com/double/slash"),
        ("mailto:someone@example.com", None),
        ("javascript:void(0)", None),
        ("http://[bad", None),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_relative():
    assert normalize("../img/../p?q=1", "http://ex.com/a/b/c.html") == "http://ex.com/a/p?q=1"


GOOGLE_STYLE = """
User-agent: *
Disallow: /
Allow: /public$
Allow: /docs/
Disallow: /docs/private/
Disallow: /*.pdf$
Crawl-delay: 2

User-agent: seekrbot
User-agent: otherbot
Disallow: /no-seekr/
"""


def test_longest_match_wins_regardless_of_order():
    p = parse(GOOGLE_STYLE)
    assert p.can_fetch("googlebot", "/public")
    assert not p.can_fetch("googlebot", "/public/more")  # $ anchors the end
    assert p.can_fetch("googlebot", "/docs/intro")
    assert not p.can_fetch("googlebot", "/docs/private/x")
    assert not p.can_fetch("googlebot", "/docs/manual.pdf")  # /*.pdf$ (7) beats /docs/ (6)
    assert not p.can_fetch("googlebot", "/anything-else")


def test_allow_wins_ties():
    p = parse("User-agent: *\nDisallow: /page\nAllow: /page\n")
    assert p.can_fetch("x", "/page")


def test_specific_group_replaces_star_group():
    p = parse(GOOGLE_STYLE)
    # SeekrBot has its own group, so the '*' group's "Disallow: /" doesn't apply to it.
    assert p.can_fetch("SeekrBot/0.1 (+url)", "/anything-else")
    assert not p.can_fetch("SeekrBot/0.1", "/no-seekr/page")
    assert p.crawl_delay("SeekrBot/0.1") is None
    assert p.crawl_delay("googlebot") == 2.0


def test_empty_disallow_allows_everything_and_robots_txt_always_allowed():
    assert parse("User-agent: *\nDisallow:\n").can_fetch("x", "/anything")
    assert parse("User-agent: *\nDisallow: /\n").can_fetch("x", "/robots.txt")


def test_unreachable_robots_follows_rfc9309():
    assert RobotsPolicy.from_status(404, "").can_fetch("x", "/a")  # 4xx -> allow all
    assert not RobotsPolicy.from_status(503, "").can_fetch("x", "/a")  # 5xx -> disallow all


def test_percent_encoding_is_compared_canonically():
    p = parse("User-agent: *\nDisallow: /caf%C3%A9\n")
    assert not p.can_fetch("x", "/café")
