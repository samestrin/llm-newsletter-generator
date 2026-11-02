"""A small HTTP GET with a hard timeout, so one slow feed cannot stall a run."""

import http.client
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit


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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, _lower(response.headers), response.read())
    except urllib.error.HTTPError as error:
        # 304 and every 4xx/5xx land here; they are answers, not transport failures.
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


def _lower(headers) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()}
