import json
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from newsletter.editor import Edition
from newsletter.render import build_report
from newsletter.sources import FeedFailure, HackerNewsSource
from newsletter.sources.http import FetchError, HttpResponse, http_get

API = "https://hn.example/v0"

STORIES = {
    1: {
        "id": 1,
        "type": "story",
        "title": "Open model ships",
        "url": "https://a.example/x",
        "score": 250,
        "time": 1_780_000_000,
        "by": "pg",
    },
    2: {"id": 2, "type": "story", "title": "Low scorer", "url": "https://b.example/y", "score": 40},
    3: {
        "id": 3,
        "type": "story",
        "title": "Ask HN: tools?",
        "score": 120,
        "text": "<p>What do you use?</p><p>I use <i>vim</i>.</p>",
    },
    4: {
        "id": 4,
        "type": "job",
        "title": "We are hiring",
        "url": "https://c.example/j",
        "score": 500,
    },
    5: {"id": 5, "type": "story", "title": "Gone", "deleted": True},
}


def js(value) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(value).encode())


class FakeHn:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, url, headers, timeout):
        self.requests.append((url, dict(headers), timeout))
        answer = self.routes.get(url, HttpResponse(404))
        if isinstance(answer, Exception):
            raise answer
        return answer


def routes(ids=(1, 2, 3, 4, 5)):
    table = {f"{API}/topstories.json": js(list(ids))}
    table.update({f"{API}/item/{i}.json": js(s) for i, s in STORIES.items()})
    return table


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr("newsletter.sources.http.resolve_host", lambda host: ["93.184.215.14"])


def test_stories_under_the_threshold_are_filtered():
    hn = FakeHn(routes())
    result = HackerNewsSource(min_score=100, api_url=API, fetch=hn).collect()
    assert result.failures == []
    assert [i.title for i in result.items] == ["Open model ships", "Ask HN: tools?"]


def test_a_story_at_the_threshold_is_kept():
    result = HackerNewsSource(min_score=120, api_url=API, fetch=FakeHn(routes())).collect()
    assert [i.title for i in result.items] == ["Open model ships", "Ask HN: tools?"]


def test_item_fields():
    hn = FakeHn(routes())
    first, ask = (
        HackerNewsSource(
            min_score=100, api_url=API, fetch=hn, user_agent="newsletter-test/1", timeout=3.5
        )
        .collect()
        .items
    )
    assert first.link == "https://a.example/x"
    assert first.feed_url == f"{API}/topstories.json"
    assert first.feed_title == "Hacker News"
    assert first.published is not None and first.published.year == 2026
    assert first.summary == "Open model ships"
    assert first.rights is None
    # A story with no URL links to its discussion, and carries its text.
    assert ask.link == "https://news.ycombinator.com/item?id=3"
    assert ask.summary == "Ask HN: tools?\nWhat do you use?\nI use vim."
    assert ask.published is None
    assert hn.requests[0] == (f"{API}/topstories.json", {"User-Agent": "newsletter-test/1"}, 3.5)


def test_limit_caps_how_many_stories_are_looked_at():
    hn = FakeHn(routes())
    HackerNewsSource(min_score=0, api_url=API, fetch=hn, limit=2).collect()
    assert [r[0] for r in hn.requests] == [
        f"{API}/topstories.json",
        f"{API}/item/1.json",
        f"{API}/item/2.json",
    ]


def test_a_failed_story_is_listed_and_the_rest_load():
    table = routes()
    table[f"{API}/item/1.json"] = FetchError("timed out after 10s")
    table[f"{API}/item/2.json"] = HttpResponse(500)
    table[f"{API}/item/3.json"] = HttpResponse(200, {}, b"not json")
    table[f"{API}/item/4.json"] = js(
        {"id": 4, "type": "story", "title": "ok", "score": 999, "url": "https://c.example/"}
    )
    result = HackerNewsSource(min_score=0, api_url=API, fetch=FakeHn(table)).collect()
    assert [i.title for i in result.items] == ["ok"]
    assert result.failures == [
        FeedFailure(f"{API}/item/1.json", "timed out after 10s"),
        FeedFailure(f"{API}/item/2.json", "HTTP 500"),
        FeedFailure(f"{API}/item/3.json", "not valid JSON"),
    ]


@pytest.mark.parametrize(
    "answer, reason",
    [
        (HttpResponse(503), "HTTP 503"),
        (FetchError("timed out after 10s"), "timed out after 10s"),
        (HttpResponse(200, {}, b"{}"), "not a list of story ids"),
        (HttpResponse(200, {}, b"nope"), "not valid JSON"),
    ],
)
def test_a_failed_story_list_is_listed(answer, reason):
    hn = FakeHn({f"{API}/topstories.json": answer})
    result = HackerNewsSource(api_url=API, fetch=hn).collect()
    assert result.items == []
    assert result.failures == [FeedFailure(f"{API}/topstories.json", reason)]


def test_a_redirect_to_a_private_address_is_refused():
    hn = FakeHn({f"{API}/topstories.json": HttpResponse(302, {"location": "http://internal/x"})})
    result = HackerNewsSource(api_url=API, fetch=hn, resolve=lambda h: ["10.0.0.1"]).collect()
    assert result.failures == [
        FeedFailure(
            f"{API}/topstories.json",
            "refused: internal is a private or loopback address (10.0.0.1)",
        )
    ]


def test_a_failed_load_is_in_the_run_report():
    hn = FakeHn({f"{API}/topstories.json": HttpResponse(503)})
    result = HackerNewsSource(api_url=API, fetch=hn).collect()
    report = build_report(result.failures, Edition(stories=[]), {}, {})
    assert report.failed_sources == [FeedFailure(f"{API}/topstories.json", "HTTP 503")]
    assert not report.ok


# A real server on 127.0.0.1, fetched with the epic 01 T3 `http_get`.


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        story = {"id": 1, "type": "story", "title": "T", "score": 200, "url": "https://a.example/"}
        big = {**story, "id": 2, "text": "x" * 5000}
        routes = {
            "/v0/topstories.json": [1, 2, 3],
            "/v0/item/1.json": story,
            "/v0/item/2.json": big,
        }
        if self.path == "/v0/item/3.json":
            time.sleep(1)
        body = json.dumps(routes.get(self.path)).encode()
        self.send_response(200 if self.path in routes else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def api_url(monkeypatch):
    monkeypatch.undo()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v0"
    httpd.shutdown()
    httpd.server_close()


def test_server_timeout_and_size_cap_are_listed(api_url):
    source = HackerNewsSource(
        api_url=api_url, timeout=0.3, fetch=partial(http_get, max_bytes=1000), min_score=0
    )
    result = source.collect()
    assert [i.title for i in result.items] == ["T"]
    assert result.failures == [
        FeedFailure(f"{api_url}/item/2.json", "response larger than 1000 bytes"),
        FeedFailure(f"{api_url}/item/3.json", "timed out after 0.3s"),
    ]
