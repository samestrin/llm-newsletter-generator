"""The Hacker News source: the front-page stories from the public API above a score threshold,
one item each. Every failed load is reported, never skipped."""

import json
from datetime import UTC, datetime

from newsletter.sources.feed import Item
from newsletter.sources.html_text import html_to_text
from newsletter.sources.http import Fetch, FetchError, Resolve, follow_redirects, http_get
from newsletter.sources.rss import DEFAULT_USER_AGENT, FeedFailure, SourceResult

API_URL = "https://hacker-news.firebaseio.com/v0"
DISCUSSION_URL = "https://news.ycombinator.com/item?id={}"


class HackerNewsSource:
    def __init__(
        self,
        *,
        min_score: int = 100,
        limit: int = 30,
        api_url: str = API_URL,
        timeout: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        fetch: Fetch = http_get,
        resolve: Resolve | None = None,
    ) -> None:
        self.min_score = min_score
        self.limit = limit
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self.fetch = fetch
        self.resolve = resolve

    def collect(self) -> SourceResult:
        """The top `limit` stories scoring at least `min_score`. A failed story list or story
        is listed in `failures`; stories under the threshold are filtered, not failures."""
        result = SourceResult()
        list_url = f"{self.api_url}/topstories.json"
        try:
            ids = self._get_json(list_url)
            if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                raise FetchError("not a list of story ids")
        except FetchError as error:
            result.failures.append(FeedFailure(list_url, str(error)))
            return result

        for story_id in ids[: self.limit]:
            url = f"{self.api_url}/item/{story_id}.json"
            try:
                item = self._story(self._get_json(url), list_url)
            except FetchError as error:
                result.failures.append(FeedFailure(url, str(error)))
                continue
            if item:
                result.items.append(item)
        return result

    def _get_json(self, url: str):
        _, response = follow_redirects(
            self.fetch,
            url,
            {"User-Agent": self.user_agent},
            self.timeout,
            resolve=self.resolve,
        )
        if response.status != 200:
            raise FetchError(f"HTTP {response.status}")
        try:
            return json.loads(response.body)
        except ValueError as error:
            raise FetchError("not valid JSON") from error

    def _story(self, data, list_url: str) -> Item | None:
        """The item for a story at or over the threshold; None for anything else."""
        if (
            not isinstance(data, dict)
            or data.get("type") != "story"
            or data.get("deleted")
            or data.get("dead")
        ):
            return None
        title, score = data.get("title"), data.get("score")
        if not isinstance(title, str) or not title or not isinstance(score, int):
            return None
        if score < self.min_score:
            return None
        link = data.get("url")
        if not isinstance(link, str) or not link:
            if not isinstance(data.get("id"), int):
                raise FetchError("story has no id")
            link = DISCUSSION_URL.format(data["id"])
        text = data.get("text")
        summary = f"{title}\n{html_to_text(text)}" if isinstance(text, str) and text else title
        published = None
        if isinstance(data.get("time"), int):
            try:
                published = datetime.fromtimestamp(data["time"], UTC)
            except (OverflowError, OSError, ValueError):
                pass  # An unusable time is left empty; the story is still good.
        return Item(
            title=title,
            link=link,
            summary=summary,
            published=published,
            feed_url=list_url,
            feed_title="Hacker News",
            rights=None,
        )
