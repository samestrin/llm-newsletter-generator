"""The sources config file: a YAML file with the topic, the feeds and the limits.

    topic: open models
    sources:
      - https://example.com/feed.xml
    limits:
      per_feed: 3

It is checked in full on load, and every error names the file and the field.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_KEYS = {"topic", "sources", "limits"}
_LIMITS = {"per_feed"}


class SourcesConfigError(ValueError):
    """The config file cannot be read or is invalid; the message names the field."""


@dataclass(frozen=True)
class SourcesConfig:
    topic: str
    sources: tuple[str, ...]
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
        raise SourcesConfigError("sources: required, a non-empty list of feed URLs")
    for n, url in enumerate(sources):
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise SourcesConfigError(f"sources[{n}]: must be an http or https URL, got {url!r}")

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

    return SourcesConfig(topic.strip(), tuple(sources), per_feed)
