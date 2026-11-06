"""The sources config file: a YAML file with the topic, the sources and the limits.

    topic: open models
    sources:
      - https://example.com/feed.xml          # an RSS or Atom feed
      - feed: https://lab.example.org/atom.xml
      - page: https://example.com/article     # a single web page
      - hackernews: {min_score: 100, limit: 30}
    limits:
      per_feed: 3

A bare URL is a feed. `hackernews` may appear once, and its settings are optional.
`limits.per_feed` keeps the first N items of each feed and of Hacker News.
It is checked in full on load, and every error names the file and the field.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_KEYS = {"topic", "sources", "limits"}
_LIMITS = {"per_feed"}
_KINDS = ("feed", "page", "hackernews")
_HACKERNEWS = {"min_score": 0, "limit": 1}
"""Each Hacker News setting and its least allowed value."""


class SourcesConfigError(ValueError):
    """The config file cannot be read or is invalid; the message names the field."""


@dataclass(frozen=True)
class HackerNewsConfig:
    min_score: int = 100
    limit: int = 30


@dataclass(frozen=True)
class SourcesConfig:
    topic: str
    feeds: tuple[str, ...] = ()
    pages: tuple[str, ...] = ()
    hackernews: HackerNewsConfig | None = None
    per_feed: int | None = None


def load_config(path: Path) -> SourcesConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SourcesConfigError(f"{path}: cannot read the config file: {error}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SourcesConfigError(f"{path}: config: not valid YAML: {error}") from None
    try:
        return _validate(data)
    except SourcesConfigError as error:
        raise SourcesConfigError(f"{path}: {error}") from None


def _validate(data: Any) -> SourcesConfig:
    if not isinstance(data, dict):
        raise SourcesConfigError("config: must be a mapping with topic, sources and limits")
    for key in data:
        if key not in _KEYS:
            raise SourcesConfigError(f"{key}: unknown field (expected {', '.join(sorted(_KEYS))})")

    topic = data.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        raise SourcesConfigError("topic: required, a non-empty string")

    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SourcesConfigError("sources: required, a non-empty list of sources")
    feeds: list[str] = []
    pages: list[str] = []
    hackernews = None
    for n, entry in enumerate(sources):
        if isinstance(entry, str):
            feeds.append(_url(f"sources[{n}]", entry))
            continue
        if not isinstance(entry, dict) or len(entry) != 1 or next(iter(entry)) not in _KINDS:
            raise SourcesConfigError(
                f"sources[{n}]: must be a feed URL or one of "
                f"{', '.join(k + ':' for k in _KINDS)} with its value"
            )
        ((kind, value),) = entry.items()
        if kind == "feed":
            feeds.append(_url(f"sources[{n}].feed", value))
        elif kind == "page":
            pages.append(_url(f"sources[{n}].page", value))
        elif hackernews is not None:
            raise SourcesConfigError(f"sources[{n}]: hackernews is listed more than once")
        else:
            hackernews = _hackernews(f"sources[{n}].hackernews", value)

    limits = data.get("limits", {})
    if limits is None:
        limits = {}
    if not isinstance(limits, dict):
        raise SourcesConfigError("limits: must be a mapping")
    for key in limits:
        if key not in _LIMITS:
            raise SourcesConfigError(f"limits.{key}: unknown field (expected per_feed)")
    per_feed = limits.get("per_feed")
    if per_feed is not None and (
        not isinstance(per_feed, int) or isinstance(per_feed, bool) or per_feed < 1
    ):
        raise SourcesConfigError(
            f"limits.per_feed: must be a whole number of at least 1, got {per_feed!r}"
        )

    return SourcesConfig(topic.strip(), tuple(feeds), tuple(pages), hackernews, per_feed)


def _url(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise SourcesConfigError(f"{name}: must be an http or https URL, got {value!r}")
    return value


def _hackernews(name: str, value: Any) -> HackerNewsConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise SourcesConfigError(f"{name}: must be a mapping with min_score and limit")
    for key in value:
        if key not in _HACKERNEWS:
            raise SourcesConfigError(f"{name}.{key}: unknown field (expected limit, min_score)")
    for key, least in _HACKERNEWS.items():
        number = value.get(key)
        if number is not None and (
            not isinstance(number, int) or isinstance(number, bool) or number < least
        ):
            raise SourcesConfigError(
                f"{name}.{key}: must be a whole number of at least {least}, got {number!r}"
            )
    return HackerNewsConfig(**value)
