"""A plain issue: one fast-model summary per item, written as Markdown with a link per item.

The link is written by this code from the feed, never by the model, so every story in
the issue can be opened. A story whose summary failed stays in the issue with its link,
and the run report at the end lists every failed feed, dropped entry and model call.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from importlib.resources import files
from typing import Any

from newsletter.llm import LLMClient, LLMError
from newsletter.sources import FeedFailure, Item, RssSource

MAX_ITEM_CHARS = 6000
"""Item text past this is cut before it is sent, so one long post cannot fill the context."""

SYSTEM_PROMPT = (
    "You summarize news stories for a newsletter. Use only facts stated in the story you "
    "are given. Do not add facts, quotes, numbers or links. Reply with plain text only."
)

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>#|])")


def load_prompt(name: str) -> str:
    """The prompt template `prompts/<name>.md` that ships with the package."""
    return files("newsletter").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")


def fill(template: str, **values: str) -> str:
    """Replace each `{{ name }}` with its value, in one pass, so a value is never re-read."""

    def value(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(f"prompt needs a value for {name!r}")
        return values[name]

    return _PLACEHOLDER.sub(value, template)


@dataclass
class Story:
    item: Item
    summary: str | None = None
    error: str | None = None
    """Why the summary is missing, when the model call failed."""


@dataclass
class Issue:
    title: str
    topic: str
    issue_date: date
    stories: list[Story] = field(default_factory=list)
    failures: list[FeedFailure] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    """`LLMClient.report()`: the model, tokens and cost of every step."""

    @property
    def ok(self) -> bool:
        """True only when there are stories and nothing failed or was left out."""
        return (
            bool(self.stories)
            and not self.failures
            and not any(story.error for story in self.stories)
        )


def summarize(items: list[Item], llm: LLMClient, *, topic: str) -> list[Story]:
    """One `fast` model call per item. A failed call is kept on the story, not raised."""
    template = load_prompt("item")
    stories = []
    for number, item in enumerate(items, start=1):
        story = Story(item)
        stories.append(story)
        if not item.summary.strip():
            # Nothing to summarize: a summary from the title alone would be invented.
            continue
        prompt = fill(
            template,
            topic=topic,
            item_title=item.title,
            item_description=item.summary[:MAX_ITEM_CHARS],
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            story.summary = llm.complete(f"summarize-{number}", messages, tier="fast").strip()
        except LLMError as error:
            story.error = str(error)
    return stories


def make_issue(
    urls: list[str],
    *,
    title: str,
    topic: str,
    source: RssSource,
    llm: LLMClient,
    issue_date: date,
) -> Issue:
    """Read the feeds, summarize every item and return the issue with its run report."""
    collected = source.collect(urls)
    stories = summarize(collected.items, llm, topic=topic)
    return Issue(
        title=title,
        topic=topic,
        issue_date=issue_date,
        stories=stories,
        failures=collected.failures,
        usage=llm.report(),
    )


def render_markdown(issue: Issue) -> str:
    lines = [f"# {_one_line(issue.title)}", "", f"*{issue.issue_date.isoformat()}*", ""]
    if not issue.stories:
        lines += ["No stories this time: no feed returned an item. See the run report.", ""]
    for story in issue.stories:
        lines += _story(story)
    lines += _sources(issue.stories)
    lines += _run_report(issue)
    return "\n".join(lines).rstrip() + "\n"


def _story(story: Story) -> list[str]:
    item = story.item
    byline = _escape(item.feed_title)
    if item.published:
        byline += f" · {item.published.date().isoformat()}"
    lines = [f"## [{_escape(item.title)}](<{_url(item.link)}>)", "", f"*{byline}*", ""]
    if story.error:
        lines.append("*Summary failed; see the run report. Read the story at the link above.*")
    elif story.summary:
        lines.append(story.summary)
    else:
        lines.append("*The feed gave no text for this story.*")
    return [*lines, ""]


def _sources(stories: list[Story]) -> list[str]:
    feeds: dict[str, Item] = {}
    for story in stories:
        feeds.setdefault(story.item.feed_url, story.item)
    if not feeds:
        return []
    lines = ["## Sources", ""]
    for url, item in feeds.items():
        line = f"- [{_escape(item.feed_title)}](<{_url(url)}>)"
        if item.rights:
            line += f": {_escape(item.rights)}"
        lines.append(line)
    return [*lines, ""]


def _run_report(issue: Issue) -> list[str]:
    lines = ["## Run report", ""]
    if issue.failures:
        lines += ["Feeds and entries left out:", ""]
        lines += [f"- {_code(f.url)}: {_escape(f.reason)}" for f in issue.failures]
    else:
        lines.append("Every feed was read and every entry was used.")
    lines.append("")

    steps = issue.usage.get("steps", [])
    if steps:
        lines += [
            "| Step | Tier | Model | Tokens in | Tokens out | Cost | Result |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for step in steps:
            cost = "unknown" if step["cost"] is None else f"${step['cost']:.6f}"
            result = "ok" if step["ok"] else f"failed: {_escape(step['error'] or '')}"
            lines.append(
                f"| {step['step']} | {step['tier']} | {_escape(step['model'])} "
                f"| {step['prompt_tokens']} | {step['completion_tokens']} | {cost} | {result} |"
            )
        lines.append("")
        total = f"Total cost: ${issue.usage['total_cost']:.6f}"
        unknown = sum(1 for step in steps if step["cost"] is None)
        if unknown:
            total += f", plus a cost unknown for {unknown} of {len(steps)} steps"
        lines += [total, ""]
    else:
        lines += ["No model calls were made.", ""]
    return lines


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _escape(text: str) -> str:
    """Plain text that Markdown shows as written, on one line."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", _one_line(text))


def _url(url: str) -> str:
    """A URL safe inside `(<...>)`: no angle brackets, spaces or line breaks."""
    return (
        url.strip()
        .replace("<", "%3C")
        .replace(">", "%3E")
        .replace(" ", "%20")
        .replace("\n", "")
        .replace("\r", "")
    )


def _code(text: str) -> str:
    """Inline code that survives backticks in `text`."""
    text = _one_line(text)
    if "`" not in text:
        return f"`{text}`"
    return f"`` {text} ``"
