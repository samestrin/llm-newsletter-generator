from datetime import UTC, datetime
from pathlib import Path

import pytest

from newsletter.sources.feed import FeedError, parse_feed

FIXTURES = Path(__file__).parent / "fixtures" / "feeds"


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_rss_fixture_parses():
    feed = parse_feed(load("rss.xml"), "https://example.com/feed.xml")

    assert feed.title == "Example AI News"
    assert feed.rights == "Copyright 2026 Example Media. Headlines and links only."
    assert feed.ttl_minutes == 60
    first, second = feed.items
    assert first.title == "A new open model"
    assert first.link == "https://example.com/posts/open-model"
    assert first.summary == "The model is open & small.\nSecond paragraph."
    assert first.published == datetime(2026, 11, 2, 9, 30, tzinfo=UTC)
    assert first.feed_url == "https://example.com/feed.xml"
    assert first.feed_title == "Example AI News"
    assert first.rights == feed.rights


def test_rss_prefers_full_content_over_description():
    feed = parse_feed(load("rss.xml"), "https://example.com/feed.xml")
    assert feed.items[1].summary == "Full text\nBenchmarks are hard."
    assert feed.items[1].published is None


def test_rss_entry_without_link_is_reported_not_dropped_silently():
    feed = parse_feed(load("rss.xml"), "https://example.com/feed.xml")
    assert len(feed.items) == 2
    assert feed.problems == ["entry 'An item with no link' has no http(s) link; left out"]


def test_atom_fixture_parses():
    feed = parse_feed(load("atom.xml"), "https://lab.example.org/atom.xml")

    assert feed.title == "Example Lab Blog"
    assert feed.rights == "CC BY 4.0"
    # sy:updatePeriod daily, sy:updateFrequency 2 -> every 12 hours
    assert feed.ttl_minutes == 720
    (item,) = feed.items
    assert item.title == "Scaling notes"
    assert item.link == "https://lab.example.org/scaling"
    assert item.summary == "Notes on scaling."
    assert item.published == datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
    assert feed.problems == []


def test_malformed_feed_is_an_error():
    with pytest.raises(FeedError, match="malformed feed"):
        parse_feed(load("broken.xml"), "https://example.com/broken.xml")


def test_html_page_is_not_a_feed():
    with pytest.raises(FeedError, match="not an RSS or Atom feed"):
        parse_feed(load("not_a_feed.html"), "https://example.com/page")


def test_empty_body_is_an_error():
    with pytest.raises(FeedError):
        parse_feed(b"", "https://example.com/empty")


def rss(items: str) -> bytes:
    return (
        f'<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>{items}</channel></rss>'
    ).encode()


def test_relative_links_resolve_against_the_feed_url():
    feed = parse_feed(rss("<item><title>R</title><link>/posts/r</link></item>"), "https://ex.com/f")
    assert feed.items[0].link == "https://ex.com/posts/r"


def test_non_http_links_are_reported_not_used():
    feed = parse_feed(
        rss("<item><title>Bad</title><link>javascript:alert(1)</link></item>"), "https://ex.com/f"
    )
    assert feed.items == []
    assert feed.problems == ["entry 'Bad' has no http(s) link; left out"]


def test_plain_text_fields_are_not_treated_as_html():
    atom = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>T</title>
    <entry><title type="text">x&lt;y and y&gt;z</title><link href="https://ex.com/a"/>
    <summary type="text">if a&lt;b then</summary></entry></feed>"""
    (item,) = parse_feed(atom, "https://ex.com/f").items
    assert item.title == "x<y and y>z"
    assert item.summary == "if a<b then"
