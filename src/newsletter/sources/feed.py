"""Parse an RSS or Atom document into plain-text items, each with the link it came from."""

import calendar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import feedparser

from newsletter.sources.html_text import html_to_text

# sy:updatePeriod values in minutes (RSS 1.0 Syndication module).
_PERIOD_MINUTES = {"hourly": 60, "daily": 1440, "weekly": 10080, "monthly": 43200, "yearly": 525600}


class FeedError(Exception):
    """The document is not a feed we can read."""


@dataclass(frozen=True)
class Item:
    title: str
    link: str
    summary: str
    """Plain text, one line per paragraph."""
    published: datetime | None
    feed_url: str
    feed_title: str
    rights: str | None
    """The feed's copyright or license line, carried so the issue can credit it."""


@dataclass(frozen=True)
class ParsedFeed:
    title: str
    items: list[Item]
    rights: str | None = None
    ttl_minutes: int | None = None
    problems: list[str] = field(default_factory=list)
    """Things left out of `items`, for the run report."""


def parse_feed(body: bytes, url: str) -> ParsedFeed:
    # content-location lets feedparser resolve relative links against the feed's URL;
    # application/xml with no charset leaves the encoding to the XML declaration.
    headers = {"content-location": url, "content-type": "application/xml"}
    parsed = feedparser.parse(body, response_headers=headers)
    if not parsed.get("version"):
        raise FeedError("not an RSS or Atom feed")
    if parsed.get("bozo") and not isinstance(
        parsed.get("bozo_exception"), feedparser.CharacterEncodingOverride
    ):
        raise FeedError(f"malformed feed: {parsed.get('bozo_exception')}")

    meta = parsed.feed
    title = _text(meta, "title") or url
    rights = meta.get("rights") or None
    items: list[Item] = []
    problems: list[str] = []
    for entry in parsed.entries:
        entry_title = _text(entry, "title")
        link = entry.get("link", "")
        if urlsplit(link).scheme not in ("http", "https"):
            # Every claim needs a source a reader can open.
            problems.append(f"entry {entry_title or '(untitled)'!r} has no http(s) link; left out")
            continue
        items.append(
            Item(
                title=entry_title or link,
                link=link,
                summary=_summary(entry),
                published=_published(entry),
                feed_url=url,
                feed_title=title,
                rights=rights,
            )
        )
    return ParsedFeed(title, items, rights, _ttl_minutes(meta), problems)


def _summary(entry) -> str:
    # Prefer the full content (content:encoded, Atom <content>) over the teaser.
    for content in entry.get("content", []):
        if content.get("value"):
            return _plain(content["value"], content.get("type"))
    return _text(entry, "summary")


def _text(node, key: str) -> str:
    return _plain(node.get(key, ""), node.get(f"{key}_detail", {}).get("type"))


def _plain(value: str, content_type: str | None) -> str:
    if content_type == "text/plain":
        return "\n".join(" ".join(line.split()) for line in value.splitlines() if line.strip())
    return html_to_text(value)


def _published(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), UTC)


def _ttl_minutes(meta) -> int | None:
    """The refresh interval the feed asks for: RSS <ttl>, else sy:updatePeriod/Frequency."""
    try:
        if meta.get("ttl"):
            return max(0, int(meta["ttl"]))
        period = _PERIOD_MINUTES.get(meta.get("sy_updateperiod", "").strip().lower())
        if period:
            frequency = max(1, int(meta.get("sy_updatefrequency") or 1))
            return period // frequency
    except ValueError:
        pass
    return None
