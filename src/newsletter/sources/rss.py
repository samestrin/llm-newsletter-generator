"""The RSS and Atom source: fetch every feed, keep what parses, report every failure."""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from newsletter import __version__
from newsletter.sources.cache import CachedFeed, FeedCache
from newsletter.sources.feed import FeedError, Item, ParsedFeed, parse_feed
from newsletter.sources.http import Fetch, FetchError, check_url, follow_redirects, http_get
from newsletter.sources.robots import RobotsPolicy

DEFAULT_USER_AGENT = (
    f"newsletter/{__version__} (+https://github.com/samestrin/llm-newsletter-generator)"
)


@dataclass(frozen=True)
class FeedFailure:
    """One line of the run report: a feed (or part of one) that did not make it in."""

    url: str
    reason: str


@dataclass
class SourceResult:
    items: list[Item] = field(default_factory=list)
    failures: list[FeedFailure] = field(default_factory=list)


class RssSource:
    def __init__(
        self,
        *,
        cache: FeedCache | None = None,
        timeout: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        fetch: Fetch = http_get,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cache = cache
        self.timeout = timeout
        self.user_agent = user_agent
        self.fetch = fetch
        self.clock = clock
        self.robots = RobotsPolicy(fetch, user_agent, timeout)

    def collect(self, urls: Iterable[str]) -> SourceResult:
        """Read every feed in `urls`. A feed that fails is listed in `failures`, never skipped."""
        result = SourceResult()
        for url in urls:
            try:
                feed = self._read(url)
            except (FetchError, FeedError) as error:
                result.failures.append(FeedFailure(url, str(error)))
                continue
            result.items.extend(feed.items)
            result.failures.extend(FeedFailure(url, problem) for problem in feed.problems)
        return result

    def _read(self, url: str) -> ParsedFeed:
        check_url(url)
        cached = self.cache.get(url) if self.cache else None
        now = self.clock()
        if cached and cached.is_fresh(now):
            # The feed's own <ttl> or sy:updatePeriod says not to ask again yet.
            return parse_feed(cached.body, url, cached.final_url)

        if refusal := self.robots.refusal(url):
            raise FetchError(refusal)

        headers = {"User-Agent": self.user_agent}
        if cached and cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached and cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        final_url, response = follow_redirects(
            self.fetch, url, headers, self.timeout, before_hop=self._check_robots
        )

        if response.status == 304 and cached:
            self._store(url, CachedFeed(**{**vars(cached), "fetched_at": now}))
            return parse_feed(cached.body, url, cached.final_url)
        if response.status != 200:
            raise FetchError(f"HTTP {response.status}")

        # Parse before caching so a broken body is never kept.
        feed = parse_feed(response.body, url, final_url)
        self._store(
            url,
            CachedFeed(
                body=response.body,
                fetched_at=now,
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
                ttl_minutes=feed.ttl_minutes,
                final_url=final_url,
            ),
        )
        return feed

    def _check_robots(self, redirect_url: str) -> None:
        if refusal := self.robots.refusal(redirect_url):
            raise FetchError(f"{refusal} (redirected to {redirect_url})")

    def _store(self, url: str, entry: CachedFeed) -> None:
        if self.cache:
            self.cache.put(url, entry)
