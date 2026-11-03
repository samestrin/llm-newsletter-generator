"""The issue as Markdown and as a responsive HTML email, from templates.

Everything shown comes from the edited issue: the editor's lead, themes, intro and "why
it matters" lines, and every story's claims, flagged ones marked as flagged. Every claim
links to the source its passage was found in, and every citation in the editor's prose
links to the claim it cites. This code builds the links, never the model.

All feed and model text is untrusted. In HTML it is escaped by the template engine, in
Markdown by `markdown_text`, and a link is written only when `safe_url` accepts it: an
absolute http or https URL. Anything else is shown as text with no link.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from typing import Any
from urllib.parse import urlsplit

from jinja2 import Environment, PackageLoader, StrictUndefined

from newsletter.editor import Edition, Prose
from newsletter.render.report import RunReport
from newsletter.sources import Item

FLAG_LABELS = {
    "unsupported": "Flagged: unsupported, the source does not support this claim",
    "unchecked": "Flagged: unchecked, this claim could not be checked",
}

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~!@])")
# What GFM turns into a link in plain text: `scheme://`, `www.` and an email address
# (whose `@` is escaped above). An escaped `:` or `.` breaks the match and still shows
# the same character.
_AUTOLINK = re.compile(r"(?i)(:(?=//)|(?<=\bwww)\.)")


@dataclass
class Newsletter:
    """One issue, ready to render."""

    title: str
    issue_date: date
    edition: Edition
    stories: Mapping[str, list[Item]]
    """Each story id's source items, the first one the story's own link."""
    report: RunReport


@dataclass(frozen=True)
class Citation:
    number: int
    url: str | None


@dataclass(frozen=True)
class ProseView:
    text: str
    cites: list[Citation]


@dataclass(frozen=True)
class ClaimView:
    number: int
    text: str
    url: str | None
    flag: str | None
    """The flag label, or None for a supported claim."""


@dataclass(frozen=True)
class SourceView:
    title: str
    url: str | None
    feed: str


@dataclass
class StoryView:
    title: str
    url: str | None
    lead: bool
    byline: str
    claims: list[ClaimView] = field(default_factory=list)
    why: ProseView | None = None
    also: list[SourceView] = field(default_factory=list)
    """The story's other sources."""


def safe_url(url: str) -> str | None:
    """`url` if it is an absolute http or https URL with a host, else None.

    A URL with whitespace or control characters is refused outright: browsers drop some
    of them, so `java\\tscript:` would run as `javascript:`.
    """
    if not url or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host:
        return None
    return url


def render_markdown(newsletter: Newsletter) -> str:
    return _environments()[1].get_template("issue.md.j2").render(_context(newsletter))


def render_html(newsletter: Newsletter) -> str:
    return _environments()[0].get_template("issue.html.j2").render(_context(newsletter))


def _context(newsletter: Newsletter) -> dict[str, Any]:
    edition, report = newsletter.edition, newsletter.report
    # Claims are numbered in the order they are shown, so a citation can point to one.
    numbers: dict[str, ClaimView] = {}
    shown = []
    for story in edition.stories:
        if not story.claims:
            continue  # In the report as a dropped story.
        items = newsletter.stories.get(story.story_id) or []
        first = items[0] if items else None
        view = StoryView(
            title=first.title if first else story.story_id,
            url=safe_url(first.link) if first else None,
            lead=story.story_id == edition.lead,
            byline=_byline(items),
            also=[SourceView(i.title, safe_url(i.link), i.feed_title) for i in items[1:]],
        )
        for checked in story.claims:
            # A status with no label of its own is still flagged, never shown as fact.
            flag = FLAG_LABELS.get(checked.status, f"Flagged: {checked.status}")
            claim = ClaimView(
                len(numbers) + 1,
                checked.claim.text,
                safe_url(checked.claim.source_url),
                flag if checked.flagged else None,
            )
            numbers[checked.id] = claim
            view.claims.append(claim)
        shown.append((story.story_id, view))

    def cites(claim_ids: list[str]) -> list[Citation]:
        return [Citation(numbers[i].number, numbers[i].url) for i in claim_ids if i in numbers]

    def prose(p: Prose | None) -> ProseView | None:
        return None if p is None else ProseView(p.text, cites(p.claim_ids))

    for story_id, view in shown:
        view.why = prose(edition.why_it_matters.get(story_id))

    return {
        "title": newsletter.title,
        "date": newsletter.issue_date.isoformat(),
        "editor_failed": edition.error is not None,
        "intro": prose(edition.intro),
        "themes": [(theme.name, cites(theme.claim_ids)) for theme in edition.themes],
        "stories": [view for _, view in shown],
        "report": report,
        "dropped_stories": [(d, safe_url(d.link)) for d in report.dropped_stories],
        "steps": [_step_row(step) for step in report.steps],
        "total": _total(report),
    }


def _byline(items: list[Item]) -> str:
    feeds: dict[str, str | None] = {}
    for item in items:
        feeds.setdefault(item.feed_title, item.rights)
    parts = [f"{feed} ({rights})" if rights else feed for feed, rights in feeds.items()]
    published = next((i.published for i in items if i.published), None)
    if published:
        parts.append(published.date().isoformat())
    return " · ".join(parts)


def _step_row(step: Mapping[str, Any]) -> dict[str, Any]:
    cost = step.get("cost")
    return {
        "step": step["step"],
        "tier": step["tier"],
        "model": step["model"],
        "tokens_in": step.get("prompt_tokens", 0),
        "tokens_out": step.get("completion_tokens", 0),
        "cost": "unknown" if cost is None else f"${cost:.6f}",
        "result": "ok" if step.get("ok") else f"failed: {step.get('error') or 'unknown error'}",
    }


def _total(report: RunReport) -> str:
    total = f"Total cost: ${report.total_cost:.6f}"
    if report.unknown_costs:
        total += f", plus a cost unknown for {report.unknown_costs} of {len(report.steps)} steps"
    return total


# Markdown escaping. Every `{{ value }}` in the Markdown template goes through
# `_markdown_finalize`, so text is escaped unless a filter below built it as `_Raw`.


class _Raw(str):
    """Markdown built by this module, written as it is."""


def markdown_text(text: str) -> str:
    """Plain text that Markdown shows as written, on one line, with no link in it."""
    text = _MARKDOWN_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))
    return _AUTOLINK.sub(r"\\\1", text)


def _markdown_finalize(value: Any) -> str:
    return value if isinstance(value, _Raw) else markdown_text(value)


def _markdown_url(url: str) -> _Raw:
    """A link destination for `[text](...)`; `url` must have passed `safe_url`."""
    if safe_url(url) is None:
        raise ValueError("only a safe URL can be linked")
    return _Raw("<" + url.replace("<", "%3C").replace(">", "%3E") + ">")


def _markdown_code(text: str) -> _Raw:
    """Inline code, which no Markdown viewer turns into a link."""
    text = " ".join(str(text).split())
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    return _Raw(f"{fence} {text} {fence}" if longest else f"{fence}{text}{fence}")


@cache
def _environments() -> tuple[Environment, Environment]:
    """The HTML environment, which escapes every value, and the Markdown one."""
    common: dict[str, Any] = {
        "loader": PackageLoader("newsletter", "render/templates"),
        "undefined": StrictUndefined,
        "trim_blocks": True,
        "lstrip_blocks": True,
        "keep_trailing_newline": True,
    }
    html = Environment(autoescape=True, **common)
    markdown = Environment(autoescape=False, finalize=_markdown_finalize, **common)
    markdown.filters["url"] = _markdown_url
    markdown.filters["code"] = _markdown_code
    # Only for a macro's output: each value in it was escaped when the macro wrote it.
    markdown.filters["markdown"] = _Raw
    return html, markdown
