"""A small HTTP GET with a hard timeout, so one slow feed cannot stall a run."""

import http.client
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urljoin, urlsplit

REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class FetchError(Exception):
    """The request never produced an HTTP response: timeout, refused, bad URL."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    """Header names are lowercase."""
    body: bytes = b""


class Fetch(Protocol):
    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse: ...


def http_get(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """GET `url`. Any HTTP status is a response; only transport problems raise FetchError."""
    check_url(url)
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return HttpResponse(response.status, _lower(response.headers), response.read())
    except urllib.error.HTTPError as error:
        # Redirects, 304 and every 4xx/5xx land here; they are answers, not transport failures.
        with error:
            return HttpResponse(error.code, _lower(error.headers), error.read())
    except http.client.HTTPException as error:
        # The server broke off mid-response, e.g. a body shorter than Content-Length.
        raise FetchError(f"incomplete response: {error!r}") from error
    except TimeoutError as error:
        raise FetchError(f"timed out after {timeout:g}s") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise FetchError(f"timed out after {timeout:g}s") from error
        raise FetchError(str(error.reason)) from error
    except OSError as error:
        raise FetchError(str(error)) from error


def follow_redirects(
    fetch: Fetch,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    *,
    before_hop: Callable[[str], None] | None = None,
    limit: int = 5,
) -> tuple[str, HttpResponse]:
    """GET `url`, following up to `limit` redirects; return the final URL and its response.

    Every hop must be an http(s) URL, and `before_hop(next_url)` may raise FetchError to
    refuse one (for example, when that host's robots.txt disallows it).
    """
    hops = 0
    while True:
        response = fetch(url, headers, timeout)
        location = response.headers.get("location")
        if response.status not in REDIRECT_STATUSES or not location:
            return url, response
        if hops == limit:
            raise FetchError("too many redirects")
        hops += 1
        url = urljoin(url, location)
        check_url(url)
        if before_hop:
            before_hop(url)


def check_url(url: str) -> None:
    """Raise FetchError unless `url` is an http(s) URL with a host."""
    try:
        parts = urlsplit(url)
    except ValueError as error:
        raise FetchError(f"bad URL: {error}") from error
    if parts.scheme not in ("http", "https"):
        raise FetchError(f"only http and https URLs are fetched, not {url!r}")
    if not parts.hostname:
        raise FetchError(f"bad URL: no host in {url!r}")


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Hand redirects back to the caller instead of following them unchecked."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirects)


def _lower(headers) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()}
