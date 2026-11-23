"""Is this address a feed a run can read? Answered before the run, with no model call.

The check reads the feed the way a run does (`RssSource`'s robots.txt rule, user agent,
timeout, size cap and redirect rules, http and https only), and refuses a private or
loopback address up front, the first address included: a visitor's address is never
fetched on the server's own network. A feed passes when it answers, parses as RSS or
Atom and has at least one item; otherwise `error` says why in one plain sentence.
"""

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from newsletter.sources.feed import FeedError, parse_feed
from newsletter.sources.http import (
    MAX_BYTES,
    BadUrl,
    Fetch,
    FetchError,
    Refused,
    Resolve,
    TooLarge,
    check_public,
    check_url,
    follow_redirects,
)
from newsletter.sources.robots import RobotsPolicy
from newsletter.sources.rss import DEFAULT_USER_AGENT

UNREACHABLE = "This address did not answer"
NOT_A_FEED = "This is a web page, not a feed"
NO_ITEMS = "This feed has no items"
ROBOTS = "The site asks not to be read"
PRIVATE = "This is a private or local address, which is not read"
NOT_WEB = "Only http and https addresses can be read"
TOO_LARGE = f"This feed is larger than {MAX_BYTES // (1024 * 1024)} MB"
BROKEN = "This feed is broken and cannot be read"
HTTP_ERROR = "This address answered with an error (HTTP {status})"

log = logging.getLogger(__name__)

TIMEOUT = 10.0
"""Seconds each request of a check may take, as a run's feed fetch."""


@dataclass(frozen=True)
class FeedCheck:
    ok: bool
    title: str | None
    items: int
    error: str | None


class _Failed(Exception):
    """The sentence a failed check answers."""


def check_feed(
    url: str,
    *,
    fetch: Fetch,
    resolve: Resolve,
    timeout: float = TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FeedCheck:
    """Fetch and parse `url` as a run would. Never raises for anything the address does."""
    try:
        title, items = _read(url, fetch, resolve, timeout, user_agent)
    except _Failed as failed:
        return FeedCheck(ok=False, title=None, items=0, error=str(failed))
    except Exception:
        # The address is the caller's: whatever it does is an answer, never a 500.
        log.exception("a feed check failed unexpectedly")
        return FeedCheck(ok=False, title=None, items=0, error=UNREACHABLE)
    if not items:
        return FeedCheck(ok=False, title=None, items=0, error=NO_ITEMS)
    return FeedCheck(ok=True, title=title, items=items, error=None)


def _read(
    url: str, fetch: Fetch, resolve: Resolve, timeout: float, user_agent: str
) -> tuple[str, int]:
    try:
        check_url(url)
        check_public(url, resolve)
    except FetchError as error:
        raise _Failed(_sentence(error)) from None
    robots = RobotsPolicy(fetch, user_agent, timeout, resolve)

    def allowed(target: str) -> None:
        refusal = robots.refusal(target)
        if refusal == "disallowed by robots.txt":
            raise _Failed(ROBOTS)
        if refusal:  # robots.txt itself could not be read.
            raise _Failed(UNREACHABLE)

    allowed(url)
    try:
        final_url, response = follow_redirects(
            fetch,
            url,
            {"User-Agent": user_agent},
            timeout,
            before_hop=allowed,
            resolve=resolve,
        )
    except FetchError as error:
        raise _Failed(_sentence(error)) from None
    if response.status != 200:
        raise _Failed(HTTP_ERROR.format(status=response.status))
    try:
        feed = parse_feed(response.body, url, final_url)
    except FeedError as error:
        raise _Failed(BROKEN if str(error).startswith("malformed") else NOT_A_FEED) from None
    return feed.title, len(feed.items)


def _sentence(error: FetchError) -> str:
    if isinstance(error, Refused):
        return PRIVATE
    if isinstance(error, BadUrl):
        return NOT_WEB
    if isinstance(error, TooLarge):
        return TOO_LARGE
    return UNREACHABLE


class RateLimit:
    """At most `limit` calls in any `window` seconds, for every caller together."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window
        self.clock = clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def take(self) -> int | None:
        """Count a call: None when it may go ahead, else the whole seconds to wait."""
        with self._lock:
            now = self.clock()
            while self._calls and self._calls[0] <= now - self.window:
                self._calls.popleft()
            if len(self._calls) >= self.limit:
                return max(1, math.ceil(self._calls[0] + self.window - now))
            self._calls.append(now)
            return None
