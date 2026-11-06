import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from newsletter.editor import Edition
from newsletter.render import build_report
from newsletter.sources import FeedFailure, PageSource
from newsletter.sources.http import FetchError, HttpResponse, http_get

PAGES = Path(__file__).parent / "fixtures" / "pages"
PAGE = "https://news.example/2026/open-model"
ROBOTS = "https://news.example/robots.txt"


def page(name: str) -> bytes:
    return (PAGES / name).read_bytes()


def html(body: bytes, content_type: str = "text/html; charset=utf-8") -> HttpResponse:
    return HttpResponse(200, {"content-type": content_type}, body)


class FakeWeb:
    """Answers GETs from a dict of URL -> response (or exception) and records each request."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, url, headers, timeout):
        self.requests.append((url, dict(headers), timeout))
        answer = self.routes.get(url, HttpResponse(404))
        if isinstance(answer, Exception):
            raise answer
        return answer

    def fetched(self, url):
        return [r for r in self.requests if r[0] == url]


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """The hosts here are made up: resolve every one to a public address, never ask real DNS."""
    monkeypatch.setattr("newsletter.sources.http.resolve_host", lambda host: ["93.184.215.14"])


def test_a_page_becomes_one_item_with_its_main_text():
    web = FakeWeb({PAGE: html(page("article.html"))})
    result = PageSource(fetch=web, user_agent="newsletter-test/1", timeout=3.5).collect([PAGE])
    assert result.failures == []
    [item] = result.items
    assert item.title == "Open model ships & tops the board | Example News"
    assert item.link == PAGE
    assert item.summary.startswith("Open model ships\nAn open-weights model was released")
    assert "Subscribe" not in item.summary
    assert item.feed_url == PAGE
    assert item.feed_title == "news.example"
    assert item.published is None
    assert item.rights is None
    assert web.fetched(PAGE) == [(PAGE, {"User-Agent": "newsletter-test/1"}, 3.5)]


def test_every_fixture_page_extracts_text():
    names = ["article.html", "main.html", "blog.html", "two-articles.html", "no-title.html"]
    urls = [f"https://news.example/{name}" for name in names]
    web = FakeWeb({url: html(page(name)) for url, name in zip(urls, names, strict=True)})
    result = PageSource(fetch=web).collect(urls)
    assert result.failures == []
    assert [item.link for item in result.items] == urls
    assert all(item.summary and item.title for item in result.items)


def test_a_page_that_robots_forbids_is_listed_not_fetched():
    web = FakeWeb({ROBOTS: html(b"User-agent: *\nDisallow: /2026/\n", "text/plain"), PAGE: html(page("article.html"))})
    result = PageSource(fetch=web).collect([PAGE])
    assert result.items == []
    assert result.failures == [FeedFailure(PAGE, "disallowed by robots.txt")]
    assert web.fetched(PAGE) == []


def test_a_redirect_to_a_page_robots_forbids_is_refused():
    final = "https://other.example/story"
    web = FakeWeb(
        {
            PAGE: HttpResponse(301, {"location": final}),
            "https://other.example/robots.txt": html(b"User-agent: *\nDisallow: /\n", "text/plain"),
            final: html(page("article.html")),
        }
    )
    result = PageSource(fetch=web).collect([PAGE])
    assert result.failures == [
        FeedFailure(PAGE, f"disallowed by robots.txt (redirected to {final})")
    ]
    assert web.fetched(final) == []


def test_a_redirected_page_links_to_where_it_was_served_from():
    final = "https://www.news.example/story"
    web = FakeWeb({PAGE: HttpResponse(302, {"location": final}), final: html(page("main.html"))})
    [item] = PageSource(fetch=web).collect([PAGE]).items
    assert item.link == final
    assert item.feed_url == PAGE
    assert item.feed_title == "www.news.example"


def test_a_redirect_to_a_private_address_is_refused():
    web = FakeWeb({PAGE: HttpResponse(302, {"location": "http://internal.example/admin"})})
    result = PageSource(fetch=web, resolve=lambda host: ["192.168.1.1"]).collect([PAGE])
    assert result.failures == [
        FeedFailure(PAGE, "refused: internal.example is a private or loopback address (192.168.1.1)")
    ]
    assert web.fetched("http://internal.example/admin") == []


@pytest.mark.parametrize(
    "answer, reason",
    [
        (HttpResponse(404), "HTTP 404"),
        (HttpResponse(500), "HTTP 500"),
        (FetchError("timed out after 10s"), "timed out after 10s"),
        (html(b"%PDF-1.7", "application/pdf"), "not an HTML page (application/pdf)"),
        (html(b"<html><body><script>x()</script></body></html>"), "no text found on the page"),
    ],
)
def test_a_failed_load_is_listed_and_the_rest_still_load(answer, reason):
    good = "https://news.example/good"
    web = FakeWeb({PAGE: answer, good: html(page("main.html"))})
    result = PageSource(fetch=web).collect([PAGE, good])
    assert result.failures == [FeedFailure(PAGE, reason)]
    assert [item.link for item in result.items] == [good]


def test_a_malformed_url_is_listed():
    result = PageSource(fetch=FakeWeb({})).collect(["ftp://news.example/page"])
    assert [f.reason for f in result.failures] == [
        "only http and https URLs are fetched, not 'ftp://news.example/page'"
    ]


def test_the_charset_in_the_header_is_used():
    body = "<p>Café crème</p>".encode("latin-1")
    web = FakeWeb({PAGE: html(body, "text/html; charset=ISO-8859-1")})
    assert PageSource(fetch=web).collect([PAGE]).items[0].summary == "Café crème"


def test_an_unknown_charset_falls_back_to_utf8():
    web = FakeWeb({PAGE: html("<p>Café</p>".encode(), "text/html; charset=nonsense")})
    assert PageSource(fetch=web).collect([PAGE]).items[0].summary == "Café"


def test_a_page_with_no_content_type_is_read_as_html():
    web = FakeWeb({PAGE: HttpResponse(200, {}, page("main.html"))})
    assert PageSource(fetch=web).collect([PAGE]).items[0].title == "Release notes"


def test_a_failed_load_is_listed_in_the_run_report():
    web = FakeWeb({PAGE: HttpResponse(503)})
    result = PageSource(fetch=web).collect([PAGE])
    report = build_report(result.failures, Edition(stories=[]), {}, {})
    assert report.failed_sources == [FeedFailure(PAGE, "HTTP 503")]
    assert not report.ok


# A real server on 127.0.0.1, fetched with the epic 01 T3 `http_get`.


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        port = self.server.server_address[1]
        routes = {
            "/robots.txt": (200, b"User-agent: *\nDisallow: /private/\n", {}),
            "/article": (200, page("article.html"), {}),
            "/private/page": (200, page("article.html"), {}),
            "/big": (200, b"<p>" + b"x" * 5000 + b"</p>", {}),
            "/to-loopback": (302, b"", {"Location": f"http://localhost:{port}/article"}),
        }
        if self.path == "/slow":
            time.sleep(1)
        status, body, headers = routes.get(self.path, (404, b"", {}))
        self.server.paths.append(self.path)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.undo()  # Resolve hosts for real: the redirect below goes to localhost.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    httpd.paths = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def base(httpd) -> str:
    return f"http://127.0.0.1:{httpd.server_address[1]}"


def test_server_page_is_fetched_and_extracted(server):
    result = PageSource().collect([f"{base(server)}/article"])
    assert result.failures == []
    assert result.items[0].summary.startswith("Open model ships\n")
    assert server.paths == ["/robots.txt", "/article"]


def test_server_timeout_is_listed(server):
    url = f"{base(server)}/slow"
    result = PageSource(timeout=0.2).collect([url])
    assert result.failures == [FeedFailure(url, "timed out after 0.2s")]


def test_server_response_over_the_size_cap_is_listed(server):
    url = f"{base(server)}/big"
    result = PageSource(fetch=partial(http_get, max_bytes=1000)).collect([url])
    assert result.failures == [FeedFailure(url, "response larger than 1000 bytes")]


def test_server_redirect_to_loopback_is_refused(server):
    url = f"{base(server)}/to-loopback"
    result = PageSource().collect([url])
    [failure] = result.failures
    assert failure.url == url
    assert failure.reason.startswith("refused: localhost is a private or loopback address")
    assert "/article" not in server.paths


def test_server_page_robots_forbids_is_listed_not_fetched(server):
    url = f"{base(server)}/private/page"
    result = PageSource().collect([url])
    assert result.failures == [FeedFailure(url, "disallowed by robots.txt")]
    assert server.paths == ["/robots.txt"]
