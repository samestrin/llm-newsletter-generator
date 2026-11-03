"""robots.txt checks, following RFC 9309 for the cases where the file cannot be read."""

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from newsletter.sources.http import Fetch, FetchError, follow_redirects


class RobotsPolicy:
    """Answers "may we fetch this URL?", reading each host's robots.txt once."""

    def __init__(self, fetch: Fetch, user_agent: str, timeout: float) -> None:
        self.fetch = fetch
        self.user_agent = user_agent
        self.timeout = timeout
        self._hosts: dict[str, RobotFileParser | str] = {}

    def refusal(self, url: str) -> str | None:
        """None when `url` may be fetched; otherwise why it may not."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._hosts:
            self._hosts[origin] = self._load(f"{origin}/robots.txt")
        rules = self._hosts[origin]
        if isinstance(rules, str):
            return rules
        # RobotFileParser matches on the product token, so drop any "/version".
        if not rules.can_fetch(self.user_agent.split("/")[0], url):
            return "disallowed by robots.txt"
        return None

    def _load(self, robots_url: str) -> RobotFileParser | str:
        rules = RobotFileParser(robots_url)
        try:
            # RFC 9309 asks crawlers to follow at least five redirects for robots.txt.
            _, response = follow_redirects(
                self.fetch, robots_url, {"User-Agent": self.user_agent}, self.timeout
            )
        except FetchError as error:
            # Unreachable: RFC 9309 says assume everything is disallowed.
            return f"robots.txt unavailable ({error})"
        if response.status >= 500:
            return f"robots.txt unavailable (HTTP {response.status})"
        if response.status >= 400:
            rules.parse([])  # No robots.txt: everything is allowed.
        else:
            rules.parse(response.body.decode("utf-8", errors="replace").splitlines())
        return rules
