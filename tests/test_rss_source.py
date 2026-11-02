from pathlib import Path

from newsletter.sources import FeedCache, FeedFailure, RssSource
from newsletter.sources.http import FetchError, HttpResponse

FIXTURES = Path(__file__).parent / "fixtures" / "feeds"
RSS = "https://example.com/feed.xml"
ATOM = "https://lab.example.org/atom.xml"


def ok(body: bytes, **headers: str) -> HttpResponse:
    return HttpResponse(status=200, headers=headers, body=body)


def status(code: int) -> HttpResponse:
    return HttpResponse(status=code, headers={}, body=b"")


class FakeWeb:
    """Answers GETs from a dict of URL -> response (or exception) and records each request."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, url, headers, timeout):
        self.requests.append((url, dict(headers), timeout))
        answer = self.routes.get(url, status(404))
        if isinstance(answer, Exception):
            raise answer
        return answer

    def fetched(self, url):
        return [r for r in self.requests if r[0] == url]


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_fixture_feeds_parse_and_a_bad_feed_is_reported():
    web = FakeWeb(
        {
            RSS: ok(fixture("rss.xml")),
            ATOM: ok(fixture("atom.xml")),
            "https://example.com/broken.xml": ok(fixture("broken.xml")),
            "https://example.com/gone.xml": status(410),
            "https://slow.example/feed": FetchError("timed out after 10s"),
        }
    )
    result = RssSource(fetch=web).collect(
        [
            RSS,
            "https://example.com/broken.xml",
            ATOM,
            "https://example.com/gone.xml",
            "https://slow.example/feed",
        ]
    )

    assert [item.link for item in result.items] == [
        "https://example.com/posts/open-model",
        "https://example.com/posts/benchmarks",
        "https://lab.example.org/scaling",
    ]
    assert [f.url for f in result.failures] == [
        RSS,  # its entry without a link
        "https://example.com/broken.xml",
        "https://example.com/gone.xml",
        "https://slow.example/feed",
    ]
    reasons = {f.url: f.reason for f in result.failures}
    assert reasons["https://example.com/broken.xml"].startswith("malformed feed")
    assert reasons["https://example.com/gone.xml"] == "HTTP 410"
    assert reasons["https://slow.example/feed"] == "timed out after 10s"


def test_problems_inside_a_good_feed_are_reported():
    result = RssSource(fetch=FakeWeb({RSS: ok(fixture("rss.xml"))})).collect([RSS])
    assert result.failures == [
        FeedFailure(RSS, "entry 'An item with no link' has no link; left out")
    ]
    assert len(result.items) == 2


def test_timeout_and_user_agent_are_sent():
    web = FakeWeb({RSS: ok(fixture("rss.xml"))})
    RssSource(fetch=web, timeout=3.5, user_agent="newsletter-test/1").collect([RSS])
    for _url, headers, timeout in web.requests:
        assert timeout == 3.5
        assert headers["User-Agent"] == "newsletter-test/1"


def test_robots_disallow_blocks_the_feed():
    web = FakeWeb(
        {
            "https://example.com/robots.txt": ok(b"User-agent: *\nDisallow: /feed.xml\n"),
            RSS: ok(fixture("rss.xml")),
        }
    )
    result = RssSource(fetch=web).collect([RSS])
    assert result.items == []
    assert result.failures == [FeedFailure(RSS, "disallowed by robots.txt")]
    assert web.fetched(RSS) == []


def test_robots_rules_for_our_agent_apply():
    robots = b"User-agent: newsletter\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    web = FakeWeb({"https://example.com/robots.txt": ok(robots), RSS: ok(fixture("rss.xml"))})
    result = RssSource(fetch=web, user_agent="newsletter/0.0.4").collect([RSS])
    assert result.failures == [FeedFailure(RSS, "disallowed by robots.txt")]


def test_missing_robots_allows_everything():
    web = FakeWeb({RSS: ok(fixture("atom.xml"))})  # robots.txt -> 404
    result = RssSource(fetch=web).collect([RSS])
    assert result.failures == []
    assert len(result.items) == 1


def test_robots_server_error_blocks_the_feed():
    web = FakeWeb({"https://example.com/robots.txt": status(503), RSS: ok(fixture("rss.xml"))})
    result = RssSource(fetch=web).collect([RSS])
    assert result.failures == [FeedFailure(RSS, "robots.txt unavailable (HTTP 503)")]
    assert web.fetched(RSS) == []


def test_robots_unreachable_blocks_the_feed():
    web = FakeWeb(
        {
            "https://example.com/robots.txt": FetchError("timed out after 10s"),
            RSS: ok(fixture("rss.xml")),
        }
    )
    result = RssSource(fetch=web).collect([RSS])
    assert result.failures == [FeedFailure(RSS, "robots.txt unavailable (timed out after 10s)")]


def test_robots_is_fetched_once_per_host():
    other = "https://example.com/other.xml"
    web = FakeWeb({RSS: ok(fixture("atom.xml")), other: ok(fixture("atom.xml"))})
    RssSource(fetch=web).collect([RSS, other])
    assert len(web.fetched("https://example.com/robots.txt")) == 1


def test_fresh_cache_is_used_without_fetching(tmp_path):
    clock = [1000.0]
    web = FakeWeb({RSS: ok(fixture("rss.xml"))})  # rss.xml has <ttl>60</ttl>
    source = RssSource(fetch=web, cache=FeedCache(tmp_path), clock=lambda: clock[0])
    source.collect([RSS])
    clock[0] += 30 * 60
    web.requests.clear()

    result = RssSource(fetch=web, cache=FeedCache(tmp_path), clock=lambda: clock[0]).collect([RSS])

    assert web.requests == []
    assert len(result.items) == 2


def test_stale_cache_sends_a_conditional_get_and_uses_304(tmp_path):
    clock = [1000.0]
    web = FakeWeb({RSS: ok(fixture("rss.xml"), etag='"v1"', **{"last-modified": "Mon, 02 Nov"})})
    RssSource(fetch=web, cache=FeedCache(tmp_path), clock=lambda: clock[0]).collect([RSS])
    clock[0] += 61 * 60
    web.routes[RSS] = status(304)

    result = RssSource(fetch=web, cache=FeedCache(tmp_path), clock=lambda: clock[0]).collect([RSS])

    _url, headers, _timeout = web.fetched(RSS)[-1]
    assert headers["If-None-Match"] == '"v1"'
    assert headers["If-Modified-Since"] == "Mon, 02 Nov"
    assert len(result.items) == 2
    # A 304 restarts the feed's TTL.
    assert FeedCache(tmp_path).get(RSS).fetched_at == clock[0]


def test_304_without_a_cached_copy_is_a_failure(tmp_path):
    web = FakeWeb({RSS: status(304)})
    result = RssSource(fetch=web, cache=FeedCache(tmp_path)).collect([RSS])
    assert result.failures == [FeedFailure(RSS, "HTTP 304")]


def test_a_bad_body_is_not_cached(tmp_path):
    web = FakeWeb({RSS: ok(fixture("broken.xml"))})
    RssSource(fetch=web, cache=FeedCache(tmp_path)).collect([RSS])
    assert FeedCache(tmp_path).get(RSS) is None


def test_works_without_a_cache():
    web = FakeWeb({RSS: ok(fixture("rss.xml"))})
    source = RssSource(fetch=web)
    source.collect([RSS])
    source.collect([RSS])
    assert len(web.fetched(RSS)) == 2
