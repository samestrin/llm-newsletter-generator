from newsletter.sources.cache import CachedFeed, FeedCache


def test_round_trip(tmp_path):
    cache = FeedCache(tmp_path)
    entry = CachedFeed(
        body=b"<rss/>",
        etag='"abc"',
        last_modified="Mon, 02 Nov 2026 09:30:00 GMT",
        fetched_at=1000.0,
        ttl_minutes=60,
    )
    cache.put("https://example.com/feed.xml", entry)
    assert cache.get("https://example.com/feed.xml") == entry


def test_survives_a_new_instance(tmp_path):
    FeedCache(tmp_path).put("https://a.example/f", CachedFeed(body=b"x", fetched_at=1.0))
    assert FeedCache(tmp_path).get("https://a.example/f").body == b"x"


def test_miss_returns_none(tmp_path):
    assert FeedCache(tmp_path).get("https://example.com/nothing") is None


def test_urls_do_not_collide(tmp_path):
    cache = FeedCache(tmp_path)
    cache.put("https://a.example/feed", CachedFeed(body=b"a", fetched_at=1.0))
    cache.put("https://b.example/feed", CachedFeed(body=b"b", fetched_at=1.0))
    assert cache.get("https://a.example/feed").body == b"a"
    assert cache.get("https://b.example/feed").body == b"b"


def test_corrupt_entry_is_a_miss(tmp_path):
    cache = FeedCache(tmp_path)
    cache.put("https://example.com/feed", CachedFeed(body=b"x", fetched_at=1.0))
    for meta in tmp_path.glob("*.json"):
        meta.write_text("{not json")
    assert cache.get("https://example.com/feed") is None


def test_is_fresh_within_ttl():
    entry = CachedFeed(body=b"x", fetched_at=1000.0, ttl_minutes=60)
    assert entry.is_fresh(now=1000.0 + 59 * 60)
    assert not entry.is_fresh(now=1000.0 + 60 * 60)


def test_without_ttl_is_never_fresh():
    assert not CachedFeed(body=b"x", fetched_at=1000.0).is_fresh(now=1000.0)
