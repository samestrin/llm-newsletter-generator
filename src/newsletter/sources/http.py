"""A small HTTP GET with a hard timeout and a size cap, so one slow or huge source cannot
stall a run, and a redirect follower that never lands on a private or loopback address."""

import http.client
import ipaddress
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urljoin, urlsplit

REDIRECT_STATUSES = {301, 302, 303, 307, 308}

MAX_BYTES = 10 * 1024 * 1024
"""The largest body `http_get` reads. A feed or an article page is far smaller."""


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


Resolve = Callable[[str], list[str]]
"""Host name -> its IP addresses; raises FetchError when it has none."""


def http_get(
    url: str, headers: Mapping[str, str], timeout: float, *, max_bytes: int = MAX_BYTES
) -> HttpResponse:
    """GET `url`. Any HTTP status is a response; only transport problems, and a body over
    `max_bytes`, raise FetchError."""
    check_url(url)
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            body = _read(response, max_bytes)
            return HttpResponse(response.status, _lower(response.headers), body)
    except urllib.error.HTTPError as error:
        # Redirects, 304 and every 4xx/5xx land here; they are answers, not transport failures.
        with error:
            return HttpResponse(error.code, _lower(error.headers), _read(error, max_bytes))
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
    resolve: Resolve | None = None,
) -> tuple[str, HttpResponse]:
    """GET `url`, following up to `limit` redirects; return the final URL and its response.

    Every hop must be an http(s) URL whose host resolves (with `resolve`, default
    `resolve_host`) only to public addresses, and `before_hop(next_url)` may raise
    FetchError to refuse one (for example, when that host's robots.txt disallows it).
    `url` itself is not address-checked: the user chose it, and may mean a local server.
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
        check_public(url, resolve or resolve_host)
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


def check_public(url: str, resolve: Resolve) -> None:
    """Raise FetchError unless every address `url`'s host resolves to is a public one."""
    host = urlsplit(url).hostname or ""
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        addresses = [ipaddress.ip_address(a.split("%")[0]) for a in resolve(host)]
    for address in addresses:
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global or address.is_multicast:
            raise FetchError(f"refused: {host} is a private or loopback address ({address})")


def resolve_host(host: str) -> list[str]:
    """Every IP address `host` resolves to."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError) as error:
        raise FetchError(f"cannot resolve {host}: {error}") from error
    return [str(info[4][0]) for info in infos]


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Hand redirects back to the caller instead of following them unchecked."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirects)


def _read(response, max_bytes: int) -> bytes:
    """The body, or FetchError once it passes `max_bytes`, read no further than one byte over."""
    length = response.headers.get("Content-Length")
    if length and length.isdigit() and int(length) > max_bytes:
        raise FetchError(f"response larger than {max_bytes} bytes")
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise FetchError(f"response larger than {max_bytes} bytes")
    if length and length.isdigit() and len(body) < int(length):
        # A bounded read does not raise when the server breaks off early; check by hand.
        raise FetchError(f"incomplete response: {len(body)} of {length} bytes")
    return body


def _lower(headers) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()}
