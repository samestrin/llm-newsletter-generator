import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from newsletter.sources.http import FetchError, http_get


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
