"""Link-only items: a Hacker News story, an hnrss entry or a one-line RSS teaser carries no
story text of its own, only its title, link and the listing's points and comments.

Such an item gets the linked article's main text as its passage, read with the web page
source's code (its timeout, size cap, redirect and robots.txt rules). An item whose
article cannot be read is skipped, never summarized from its title or metadata. Both are
recorded for the run report.
"""

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from newsletter.sources.feed import Item
from newsletter.sources.http import FetchError
from newsletter.sources.page import PageSource

MIN_TEXT_CHARS = 50
"""An item with less text than this of its own (title and listing metadata aside) is a
link, not a story: shorter than one sentence that states a fact, such as "Read more",
a byline or "submitted by /u/name [link] [comments]". One sentence is kept as it is."""

# The lines hnrss writes for every entry, and the like from other aggregators.
_METADATA_LINE = re.compile(
    r"^(?:article url|comments url|url|link|points|score|#\s*comments|comments)\s*:",
    re.IGNORECASE,
)
_URL_LINE = re.compile(r"^\S+://\S+$")


@dataclass(frozen=True)
class LinkedArticle:
    """One line of the run report: a link-only item whose article was read, or skipped."""

    title: str
    link: str
    status: Literal["fetched", "skipped"]
    reason: str | None
    """Why the article could not be read, when it was skipped."""


@dataclass
class Linked:
    items: list[Item] = field(default_factory=list)
    """The items to group: those with their own text, and the link-only ones that now
    carry their article's text. Skipped items are not here."""
    articles: list[LinkedArticle] = field(default_factory=list)


def own_text(item: Item) -> str:
    """The item's text without its title, bare URLs and listing metadata lines."""
    lines = []
    for line in item.summary.splitlines():
        line = line.strip()
        if not line or line == item.title.strip():
            continue
        if _METADATA_LINE.match(line) or _URL_LINE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def is_link_only(item: Item) -> bool:
    return len(own_text(item)) < MIN_TEXT_CHARS


def read_linked(items: Iterable[Item], pages: PageSource, *, keep: Collection[str] = ()) -> Linked:
    """Each link-only item with its article's main text as its summary; an item from a
    feed in `keep` (a web page already read) is left as it is."""
    linked = Linked()
    for item in items:
        if item.feed_url in keep or not is_link_only(item):
            linked.items.append(item)
            continue
        try:
            article = pages.read(item.link)
        except FetchError as error:
            linked.articles.append(LinkedArticle(item.title, item.link, "skipped", str(error)))
            continue
        linked.items.append(replace(item, link=article.link, summary=article.summary))
        linked.articles.append(LinkedArticle(item.title, article.link, "fetched", None))
    return linked
