"""Where stories come from: RSS and Atom feeds, single web pages, and Hacker News."""

from newsletter.sources.cache import FeedCache
from newsletter.sources.feed import Item
from newsletter.sources.hackernews import HackerNewsSource
from newsletter.sources.page import PageSource
from newsletter.sources.rss import FeedFailure, RssSource, SourceResult

__all__ = [
    "FeedCache",
    "FeedFailure",
    "HackerNewsSource",
    "Item",
    "PageSource",
    "RssSource",
    "SourceResult",
]
