"""An on-disk cache of feed bodies and the validators needed for conditional GETs."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedFeed:
    body: bytes
    fetched_at: float
    etag: str | None = None
    last_modified: str | None = None
    ttl_minutes: int | None = None
    """How long the feed asks readers to wait before fetching it again."""
    final_url: str | None = None
    """Where the feed was served from after redirects; relative links resolve against it."""

    def is_fresh(self, now: float) -> bool:
        return self.ttl_minutes is not None and now - self.fetched_at < self.ttl_minutes * 60


class FeedCache:
    """One `<sha256>.json` (metadata) and `<sha256>.xml` (body) pair per feed URL."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def get(self, url: str) -> CachedFeed | None:
        meta_path, body_path = self._paths(url)
        try:
            meta = json.loads(meta_path.read_text())
            meta.pop("url")  # kept on disk only so a person can tell the files apart
            return CachedFeed(body=body_path.read_bytes(), **meta)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # Missing or damaged: treat it as never cached and fetch again.
            return None

    def put(self, url: str, entry: CachedFeed) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        meta_path, body_path = self._paths(url)
        body_path.write_bytes(entry.body)
        meta = {
            "fetched_at": entry.fetched_at,
            "etag": entry.etag,
            "last_modified": entry.last_modified,
            "ttl_minutes": entry.ttl_minutes,
            "final_url": entry.final_url,
        }
        meta_path.write_text(json.dumps({"url": url, **meta}))

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode()).hexdigest()
        return self.directory / f"{key}.json", self.directory / f"{key}.xml"
