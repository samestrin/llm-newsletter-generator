import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from newsletter.sources.http import FetchError, HttpResponse, follow_redirects, http_get

BIG = b"x" * 2000


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/feed":
            body = b"<rss/>"
            self.send_response(200)
            self.send_header("ETag", '"abc"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/echo-agent":
            body = self.headers["User-Agent"].encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/not-modified":
            self.send_response(304)
            self.end_headers()
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/feed")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/truncated":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"short")
            self.wfile.flush()
            self.close_connection = True
        elif self.path == "/big":
            self.send_response(200)
            self.send_header("Content-Length", str(len(BIG)))
            self.end_headers()
            self.wfile.write(BIG)
        elif self.path == "/big-no-length":
            # No Content-Length: the body ends when the connection closes.
            self.send_response(200)
            self.end_headers()
            self.wfile.write(BIG)
            self.close_connection = True
        elif self.path == "/big-error":
            self.send_response(500)
            self.send_header("Content-Length", str(len(BIG)))
            self.end_headers()
            self.wfile.write(BIG)
        elif self.path == "/to-loopback":
            port = self.server.server_address[1]
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{port}/feed")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/to-localhost":
            port = self.server.server_address[1]
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{port}/feed")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/slow":
            time.sleep(1)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_ok_response(server):
    response = http_get(f"{server}/feed", {}, timeout=5)
    assert response.status == 200
    assert response.body == b"<rss/>"
    assert response.headers["etag"] == '"abc"'


def test_sends_request_headers(server):
    response = http_get(f"{server}/echo-agent", {"User-Agent": "newsletter-test"}, timeout=5)
    assert response.body == b"newsletter-test"


def test_not_modified_is_a_response_not_an_error(server):
    assert http_get(f"{server}/not-modified", {}, timeout=5).status == 304


def test_redirects_are_returned_not_followed(server):
    # The caller follows them, so it can check each hop's scheme and robots.txt.
    response = http_get(f"{server}/redirect", {}, timeout=5)
    assert response.status == 302
    assert response.headers["location"] == "/feed"


def test_http_error_status_is_returned(server):
    assert http_get(f"{server}/missing", {}, timeout=5).status == 404


def test_timeout_is_an_error(server):
    with pytest.raises(FetchError, match="timed out after 0.2s"):
        http_get(f"{server}/slow", {}, timeout=0.2)


def test_truncated_body_is_an_error(server):
    with pytest.raises(FetchError):
        http_get(f"{server}/truncated", {}, timeout=5)


def test_connection_refused_is_an_error():
    with pytest.raises(FetchError):
        http_get("http://127.0.0.1:9/feed", {}, timeout=2)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/feed", "not a url"])
def test_only_http_and_https_are_fetched(url):
    with pytest.raises(FetchError, match="only http and https"):
        http_get(url, {}, timeout=1)


def test_a_body_over_the_size_cap_is_an_error(server):
    with pytest.raises(FetchError, match="larger than 1000 bytes"):
        http_get(f"{server}/big", {}, timeout=5, max_bytes=1000)


def test_a_body_with_no_length_over_the_size_cap_is_an_error(server):
    with pytest.raises(FetchError, match="larger than 1000 bytes"):
        http_get(f"{server}/big-no-length", {}, timeout=5, max_bytes=1000)


def test_an_error_body_over_the_size_cap_is_an_error(server):
    with pytest.raises(FetchError, match="larger than 1000 bytes"):
        http_get(f"{server}/big-error", {}, timeout=5, max_bytes=1000)


def test_a_body_at_the_size_cap_is_read(server):
    assert http_get(f"{server}/big", {}, timeout=5, max_bytes=len(BIG)).body == BIG


def test_the_default_size_cap_allows_ordinary_pages(server):
    assert http_get(f"{server}/big-no-length", {}, timeout=5).body == BIG


@pytest.mark.parametrize("path", ["/to-loopback", "/to-localhost"])
def test_a_redirect_to_a_loopback_address_is_refused(server, path):
    # The first URL is the one the user configured; only where it redirects to is checked.
    with pytest.raises(FetchError, match="private or loopback"):
        follow_redirects(http_get, f"{server}{path}", {}, 5)


def test_a_redirect_is_followed_when_the_host_resolves_to_a_public_address(server):
    resolved = []

    def resolve(host):
        resolved.append(host)
        return ["93.184.215.14"]

    final, response = follow_redirects(http_get, f"{server}/to-localhost", {}, 5, resolve=resolve)
    assert final.endswith("/feed")
    assert response.body == b"<rss/>"
    assert resolved == ["localhost"]


def hops(routes):
    def fetch(url, headers, timeout):
        return routes[url]

    return fetch


def moved(location):
    return HttpResponse(302, {"location": location})


ADDRESSES = {
    "public.example": ["93.184.215.14"],
    "also-public.example": ["2606:2800:21f:cb07:6820:80da:af6b:8b2c"],
    "private.example": ["10.1.2.3"],
    "mixed.example": ["93.184.215.14", "192.168.0.10"],
}


def fake_resolve(host):
    if host not in ADDRESSES:
        raise FetchError(f"cannot resolve {host}")
    return ADDRESSES[host]


@pytest.mark.parametrize(
    "target",
    [
        "http://private.example/x",
        "http://mixed.example/x",
        "http://10.0.0.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/x",
        "http://[::ffff:127.0.0.1]/x",
        "http://0.0.0.0/x",
        "http://[fd00::1]/x",
    ],
)
def test_a_redirect_to_a_private_address_is_refused(target):
    fetch = hops({"https://public.example/a": moved(target)})
    with pytest.raises(FetchError, match="private or loopback"):
        follow_redirects(fetch, "https://public.example/a", {}, 5, resolve=fake_resolve)


def test_every_hop_is_resolved_and_a_late_private_hop_is_refused():
    fetch = hops(
        {
            "https://public.example/a": moved("https://also-public.example/b"),
            "https://also-public.example/b": moved("https://private.example/c"),
        }
    )
    resolved = []

    def resolve(host):
        resolved.append(host)
        return fake_resolve(host)

    with pytest.raises(FetchError, match="private.example"):
        follow_redirects(fetch, "https://public.example/a", {}, 5, resolve=resolve)
    assert resolved == ["also-public.example", "private.example"]


def test_a_redirect_to_a_host_that_does_not_resolve_is_refused():
    fetch = hops({"https://public.example/a": moved("https://nowhere.example/")})
    with pytest.raises(FetchError, match="cannot resolve"):
        follow_redirects(fetch, "https://public.example/a", {}, 5, resolve=fake_resolve)


def test_the_default_resolver_reports_an_unknown_host():
    from newsletter.sources.http import resolve_host

    with pytest.raises(FetchError, match="cannot resolve"):
        resolve_host("no-such-host.invalid")
