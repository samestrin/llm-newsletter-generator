"""The web page source: fetch each article URL, keep its main text, report every failure.

It fetches and checks robots.txt with the same code as the RSS source, so a page gets the
same timeout, size cap and redirect rules as a feed.
"""

import codecs
from collections.abc import Iterable
from email.message import Message
from urllib.parse import urlsplit

from newsletter.sources.feed import Item
from newsletter.sources.html_text import extract_article
from newsletter.sources.http import (
    Fetch,
    FetchError,
    Resolve,
    check_url,
    follow_redirects,
    http_get,
)
from newsletter.sources.robots import RobotsPolicy
from newsletter.sources.rss import DEFAULT_USER_AGENT, FeedFailure, SourceResult

_HTML_TYPES = {"text/html", "application/xhtml+xml"}


class PageSource:
    def __init__(
        self,
        *,
        timeout: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        fetch: Fetch = http_get,
        resolve: Resolve | None = None,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.fetch = fetch
        self.resolve = resolve
        self.robots = RobotsPolicy(fetch, user_agent, timeout, resolve)

    def collect(self, urls: Iterable[str]) -> SourceResult:
        """Read every page in `urls`, one item each. A page that fails is listed in
        `failures`, never skipped."""
        result = SourceResult()
        for url in urls:
            try:
                result.items.append(self._read(url))
            except FetchError as error:
                result.failures.append(FeedFailure(url, str(error)))
        return result

    def _read(self, url: str) -> Item:
        check_url(url)
        if refusal := self.robots.refusal(url):
            raise FetchError(refusal)
        final_url, response = follow_redirects(
            self.fetch,
            url,
            {"User-Agent": self.user_agent},
            self.timeout,
            before_hop=self._check_robots,
            resolve=self.resolve,
        )
        if response.status != 200:
            raise FetchError(f"HTTP {response.status}")

        content_type, charset = _content_type(response.headers.get("content-type", ""))
        if content_type and content_type not in _HTML_TYPES:
            raise FetchError(f"not an HTML page ({content_type})")
        article = extract_article(response.body.decode(charset, errors="replace"))
        if not article.text:
            raise FetchError("no text found on the page")

        host = urlsplit(final_url).hostname or final_url
        return Item(
            title=article.title or final_url,
            link=final_url,
            summary=article.text,
            published=None,
            feed_url=url,
            feed_title=host,
            rights=None,
        )

    def _check_robots(self, redirect_url: str) -> None:
        if refusal := self.robots.refusal(redirect_url):
            raise FetchError(f"{refusal} (redirected to {redirect_url})")


def _content_type(header: str) -> tuple[str, str]:
    """The media type (empty when the header is missing) and a charset Python knows."""
    if not header:
        return "", "utf-8"
    message = Message()
    message["content-type"] = header
    charset = message.get_content_charset() or "utf-8"
    try:
        codecs.lookup(charset)
    except LookupError:
        charset = "utf-8"
    return message.get_content_type(), charset
