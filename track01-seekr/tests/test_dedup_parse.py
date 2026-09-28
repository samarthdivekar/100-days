import random

from seekr.crawl.dedup import NearDupIndex, content_hash, hamming, simhash
from seekr.crawl.parse import analyze, parse_html


def _doc(rng, n=300):
    words = ["alpha", "beta", "gamma", "delta", "raptor", "enron", "ledger", "swap"]
    return " ".join(rng.choice(words) + str(rng.randint(0, 500)) for _ in range(n))


def test_content_hash_ignores_case_and_whitespace():
    assert content_hash("Hello   World!") == content_hash("hello world")


def test_simhash_near_vs_unrelated():
    rng = random.Random(1)
    a = _doc(rng)
    words = a.split()
    words[10] = "edited"
    b = " ".join(words) + " updated 3 october 2026"
    c = _doc(rng)
    assert hamming(simhash(a), simhash(b)) <= 8
    assert hamming(simhash(a), simhash(c)) > 10


def test_near_dup_index_finds_within_k_bits_only():
    idx = NearDupIndex(k=3)  # no signatures given: SimHash distance alone decides
    base = 0xDEADBEEFCAFEF00D
    idx.add(base, "orig")
    assert idx.find(base ^ 0b111) == "orig"  # 3 bits
    assert idx.find(base ^ (1 | 1 << 20 | 1 << 40 | 1 << 60)) is None  # 4 bits, one per block
    wide = NearDupIndex(k=8)
    wide.add(base, "orig")
    eight = sum(1 << b for b in (0, 7, 14, 21, 28, 35, 42, 49))  # 8 bits spread over the blocks
    assert wide.find(base ^ eight) == "orig"
    assert wide.find(base ^ eight ^ (1 << 63)) is None


def test_parse_html_text_links_and_directives():
    html = """<html><head><title> Raptor  hedges </title><meta name="robots" content="noindex">
    <link rel="canonical" href="/canon"><style>.x{}</style><script>var a = 1;</script></head>
    <body><h1>Heading</h1><p>Body &amp; text</p>
    <a href="/a#frag">A</a><a href="b?utm_source=x">B</a><a href="/a">dup</a>
    <a rel="nofollow" href="/skip">skip</a><a href="mailto:x@y.z">mail</a></body></html>"""
    page = parse_html(html, "http://ex.com/dir/page")
    assert page.title == "Raptor hedges"
    assert page.noindex and not page.nofollow
    assert page.canonical == "http://ex.com/canon"
    assert "Body & text" in page.text and "var a" not in page.text and ".x" not in page.text
    assert page.links == ["http://ex.com/a", "http://ex.com/dir/b"]


def test_shared_site_chrome_does_not_make_pages_duplicates():
    """Regression from books.toscrape.com: category pages listing different books were flagged as
    near-duplicates because a 50-link sidebar dominated their text."""
    sidebar = (
        "<aside class='side_categories'><ul>"
        + "".join(f"<li><a href='/c/{i}'>Category number {i} books</a></li>" for i in range(50))
        + "</ul></aside>"
    )
    chrome = (
        f"<header class='page-header'>All products store home</header>{sidebar}<footer>Terms privacy</footer>"
    )
    page = "<html><body>{}<h1>{}</h1><p>{}</p></body></html>"
    poetry = page.format(chrome, "Poetry", "Leaves of Grass. The Waste Land. Ariel. Howl.")
    history = page.format(chrome, "History", "SPQR. Guns Germs and Steel. The Silk Roads.")
    a_page, _, a_fp, _ = analyze(poetry, "http://x/poetry")
    b_page, _, b_fp, _ = analyze(history, "http://x/history")
    assert "Category number" not in a_page.main_text and "Category number" in a_page.text
    assert "Leaves of Grass" in a_page.main_text
    assert hamming(a_fp, b_fp) > 8  # main content differs, so not near-duplicates
    assert hamming(simhash(a_page.text), simhash(b_page.text)) <= 8  # full text alone would have collided


def test_unclosed_tags_do_not_leak_chrome_state():
    html = "<body><nav><ul><li>Home<li>About</nav><p>Real article text here</p><div class='sidebar'>ad</div>"
    page = parse_html(html, "http://x/")
    assert page.main_text == "Real article text here"


def test_minhash_estimates_jaccard_and_rejects_false_candidates():
    from seekr.crawl.dedup import fingerprint, jaccard_estimate

    rng = random.Random(4)
    a = _doc(rng, 400)
    words = a.split()
    words[50] = "edited"
    near = " ".join(words) + " updated 3 october 2026"
    far = _doc(rng, 400)
    fa, sa = fingerprint(a)
    fn, sn = fingerprint(near)
    ff, sf = fingerprint(far)
    assert jaccard_estimate(sa, sn) >= 0.8 and jaccard_estimate(sa, sf) < 0.2

    idx = NearDupIndex(k=64)  # every stored page is a SimHash candidate
    idx.add(fa, "a", sa)
    assert idx.find(ff, sf) is None and idx.rejected == 1  # verification rejects it
    assert idx.find(fn, sn) == "a"


def test_repeated_template_blocks_count_once():
    """Listing pages repeat 'In stock Add to basket' per item; as a multiset that outvotes the titles."""
    tile = "In stock Add to basket price {p} pounds star rating three"
    a = " ".join(tile.format(p=i) + f" {t}" for i, t in enumerate(["Dune", "Emma", "Ulysses", "Beloved"]))
    b = " ".join(
        tile.format(p=i + 9) + f" {t}" for i, t in enumerate(["Hamlet", "Heidi", "Walden", "Ivanhoe"])
    )
    from seekr.crawl.dedup import fingerprint, jaccard_estimate

    _, sa = fingerprint(a)
    _, sb = fingerprint(b)
    assert jaccard_estimate(sa, sb) < 0.8
