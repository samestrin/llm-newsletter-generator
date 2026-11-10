"""A static site of approved issues: an index, and one page per issue; and an issue's
Markdown and HTML email, drawn from the same issue by the same rules.

The site is public, so it is built to show nothing it should not:

- A page's path comes from the issue date and a fixed slug, never from feed text.
- Every piece of feed or model text is HTML-escaped (Markdown-escaped in the
  Markdown), and pages forbid scripts.
- Only http and https links are rendered; any other link is left out.
- The run report shows failures by feed host and count only: no error body, file
  path, key or model URL. Feeds are credited by title and host, because a feed URL
  can carry a key in its query.

A site built with an `automated-demo` approval (a demo build, which no person reviews)
carries a visible "Automated demo" label at the top of every page, and every summary is
marked unchecked, because the plain issue checks no claim. The demo Markdown and email
carry the same label and marks.

The site is written to a temp dir beside the target and swapped in only when every
page is written, so a failed build leaves the previous site as it was.
"""

import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from newsletter.archive.approval import Approval, require_archive
from newsletter.issue import Issue, Story
from newsletter.render import FLAG_LABELS, markdown_text, safe_url

LINK_SCHEMES = ("http", "https")
LOCAL_SOURCE = "a local file"
DEMO_NOTE = (
    "A model drafted this from public AI news feeds with no human review. "
    "Flagged claims are marked as flagged."
)
DEMO_LABEL = (
    '<p role="note" style="border:2px solid #b45309;padding:.5rem 1rem">'
    f"<strong>Automated demo.</strong> {DEMO_NOTE}</p>"
)

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<meta name="referrer" content="no-referrer">
<title>{title}</title>
<style>body{{max-width:42rem;margin:2rem auto;padding:0 1rem;font:1rem/1.5 sans-serif}}</style>
</head>
<body>
{body}
</body>
</html>
"""


class ArchiveError(ValueError):
    """The issues cannot be built into a site."""


def issue_path(issue: Issue) -> str:
    """The page of an issue, relative to the site root: from its date only."""
    return f"issues/{issue.issue_date.isoformat()}/index.html"


def build_site(
    issues: Iterable[Issue], site_dir: Path, *, approval: Approval | None = None
) -> None:
    """Build the site for `issues` into `site_dir`, replacing it only when complete.

    With an `approval`, it is checked first; an `automated-demo` one labels every page.
    """
    if approval is not None:
        require_archive(approval)
    demo = approval is not None and approval.automated
    issues = sorted(issues, key=lambda issue: issue.issue_date, reverse=True)
    dates = Counter(issue.issue_date for issue in issues)
    repeated = sorted(day.isoformat() for day, count in dates.items() if count > 1)
    if repeated:
        raise ArchiveError(f"more than one issue on {', '.join(repeated)}")

    site_dir = Path(site_dir)
    site_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{site_dir.name}-build-", dir=site_dir.parent))
    try:
        _write(work, "index.html", render_index(issues, demo=demo))
        for issue in issues:
            _write(work, issue_path(issue), render_issue(issue, demo=demo))
        work.chmod(0o755)
        _swap(work, site_dir)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def render_index(issues: list[Issue], *, demo: bool = False) -> str:
    lines = [DEMO_LABEL] if demo else []
    lines.append("<h1>Archive</h1>")
    if not issues:
        lines.append("<p>No issues yet.</p>")
    else:
        lines.append("<ul>")
        for issue in issues:
            day = issue.issue_date.isoformat()
            stories = _count(len(issue.stories), "story", "stories")
            lines.append(
                f'<li><a href="{issue_path(issue)}">{_text(issue.title)}</a> '
                f"<time>{day}</time> ({stories})</li>"
            )
        lines.append("</ul>")
    return _PAGE.format(title="Archive", body="\n".join(lines))


def render_issue(issue: Issue, *, demo: bool = False) -> str:
    return _issue_page(issue, demo=demo, archive_link=True)


def render_email(issue: Issue, *, demo: bool = False) -> str:
    """The issue as one self-contained HTML email: the issue page without its archive link."""
    return _issue_page(issue, demo=demo, archive_link=False)


def _issue_page(issue: Issue, *, demo: bool, archive_link: bool) -> str:
    lines = [DEMO_LABEL] if demo else []
    if archive_link:
        lines.append('<p><a href="../../index.html">Archive</a></p>')
    lines += [
        f"<h1>{_text(issue.title)}</h1>",
        f"<p><time>{issue.issue_date.isoformat()}</time></p>",
    ]
    if not issue.stories:
        lines.append("<p>No stories this time: no feed returned an item.</p>")
    for story in issue.stories:
        lines += _story(story, unchecked=demo)
    lines += _sources(issue.stories)
    lines += _run_report(issue)
    return _PAGE.format(title=_text(issue.title), body="\n".join(lines))


def _story(story: Story, *, unchecked: bool = False) -> list[str]:
    item = story.item
    title = _text(item.title)
    href = _href(item.link)
    heading = f'<a href="{href}" rel="noopener noreferrer">{title}</a>' if href else title
    byline = _text(item.feed_title)
    if item.published:
        byline += f" · {item.published.date().isoformat()}"
    lines = ["<article>", f"<h2>{heading}</h2>", f"<p><em>{byline}</em></p>"]
    if story.error:
        lines.append("<p><em>Summary failed; see the run report.</em></p>")
    elif story.summary:
        if unchecked:
            lines.append(f"<p><strong>{_text(FLAG_LABELS['unchecked'])}</strong></p>")
        lines += [f"<p>{_text(part)}</p>" for part in story.summary.splitlines() if part.strip()]
    else:
        lines.append("<p><em>The feed gave no text for this story.</em></p>")
    return [*lines, "</article>"]


def _sources(stories: list[Story]) -> list[str]:
    feeds = {}
    for story in stories:
        feeds.setdefault(story.item.feed_url, story.item)
    if not feeds:
        return []
    lines = ["<h2>Sources</h2>", "<ul>"]
    for url, item in feeds.items():
        name = _text(item.feed_title)
        home = _home(url)
        line = f'<a href="{home}" rel="noopener noreferrer">{name}</a>' if home else name
        if item.rights:
            line += f": {_text(item.rights)}"
        lines.append(f"<li>{line}</li>")
    return [*lines, "</ul>"]


def _run_report(issue: Issue) -> list[str]:
    lines = ["<h2>Run report</h2>"]
    problems = _problems(issue)
    if problems:
        lines += ["<p>Feeds, entries and summaries left out:</p>", "<ul>"]
        lines += [f"<li>{_text(host)}: {_count(count, 'problem')}</li>" for host, count in problems]
        lines.append("</ul>")
    else:
        lines.append("<p>Every feed was read and every entry was used.</p>")
    return [*lines, f"<p>{_text(_calls(issue))}</p>"]


def _problems(issue: Issue) -> list[tuple[str, int]]:
    """Failed feeds, entries and summaries, counted by feed host, most first. Only the
    host is shown: a feed URL can carry a key, and an error can name a path or a URL."""
    problems = Counter(_host(failure.url) for failure in issue.failures)
    problems.update(_host(story.item.feed_url) for story in issue.stories if story.error)
    return sorted(problems.items(), key=lambda pair: (-pair[1], pair[0]))


def _calls(issue: Issue) -> str:
    """The model calls and their cost, without a model name or an error."""
    steps = issue.usage.get("steps", [])
    if not steps:
        return "No model calls were made."
    failed = sum(1 for step in steps if not step.get("ok"))
    calls = f"{_count(len(steps), 'model call')}, {failed} failed"
    cost = f"total cost ${issue.usage.get('total_cost') or 0:.6f}"
    unknown = sum(1 for step in steps if step.get("cost") is None)
    if unknown:
        cost += f", plus a cost unknown for {unknown} of {len(steps)}"
    return f"{calls}; {cost}."


def render_markdown(issue: Issue, *, demo: bool = False) -> str:
    """The issue as Markdown, by the site's rules: feed and model text escaped, only web
    links, and the run report by feed host and count."""
    lines = [f"> **Automated demo.** {markdown_text(DEMO_NOTE)}", ""] if demo else []
    lines += [f"# {markdown_text(issue.title)}", "", f"*{issue.issue_date.isoformat()}*", ""]
    if not issue.stories:
        lines += ["No stories this time: no feed returned an item.", ""]
    for story in issue.stories:
        lines += _markdown_story(story, unchecked=demo)
    lines += _markdown_sources(issue.stories)
    lines += ["## Run report", ""]
    problems = _problems(issue)
    if problems:
        lines += ["Feeds, entries and summaries left out:", ""]
        lines += [
            f"- {markdown_text(host)}: {_count(count, 'problem')}" for host, count in problems
        ]
    else:
        lines.append("Every feed was read and every entry was used.")
    lines += ["", markdown_text(_calls(issue))]
    return "\n".join(lines) + "\n"


def _markdown_story(story: Story, *, unchecked: bool) -> list[str]:
    item = story.item
    byline = markdown_text(item.feed_title)
    if item.published:
        byline += f" · {item.published.date().isoformat()}"
    lines = [f"## {_markdown_link(item.title, item.link)}", "", f"*{byline}*", ""]
    if story.error:
        lines += ["*Summary failed; see the run report.*", ""]
    elif story.summary:
        if unchecked:
            lines += [f"**{markdown_text(FLAG_LABELS['unchecked'])}**", ""]
        for part in story.summary.splitlines():
            if part.strip():
                lines += [markdown_text(part), ""]
    else:
        lines += ["*The feed gave no text for this story.*", ""]
    return lines


def _markdown_sources(stories: list[Story]) -> list[str]:
    feeds = {}
    for story in stories:
        feeds.setdefault(story.item.feed_url, story.item)
    if not feeds:
        return []
    lines = ["## Sources", ""]
    for url, item in feeds.items():
        home = _web(url)
        line = _markdown_link(item.feed_title, f"{home[0]}://{_netloc(*home[1:])}/" if home else "")
        if item.rights:
            line += f": {markdown_text(item.rights)}"
        lines.append(f"- {line}")
    return [*lines, ""]


def _markdown_link(text: str, url: str) -> str:
    """`[text](<url>)` for an http or https URL; the text alone for anything else."""
    url = url.strip()
    if _web(url) is None or safe_url(url) is None:
        return markdown_text(text)
    return f"[{markdown_text(text)}](<{url.replace('<', '%3C').replace('>', '%3E')}>)"


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ArchiveError(f"page path {relative!r} is outside the site")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _swap(work: Path, site_dir: Path) -> None:
    """Put `work` at `site_dir`; the previous site is moved aside, then removed."""
    if not site_dir.exists():
        os.rename(work, site_dir)
        return
    old = Path(tempfile.mkdtemp(prefix=f".{site_dir.name}-old-", dir=site_dir.parent))
    previous = old / "site"
    try:
        os.rename(site_dir, previous)
        try:
            os.rename(work, site_dir)
        except BaseException:
            os.rename(previous, site_dir)
            raise
    finally:
        shutil.rmtree(old, ignore_errors=True)


def _text(text: str) -> str:
    return escape(" ".join(text.split()), quote=True)


def _href(url: str) -> str | None:
    """The escaped URL when it is an absolute http or https link, else None."""
    if _web(url) is None:
        return None
    return escape(url.strip(), quote=True)


def _home(url: str) -> str | None:
    """The scheme and host of a feed URL, without its path, query or user info."""
    web = _web(url)
    if web is None:
        return None
    scheme, hostname, port = web
    return escape(f"{scheme}://{_netloc(hostname, port)}/", quote=True)


def _host(url: str) -> str:
    web = _web(url)
    return LOCAL_SOURCE if web is None else web[1]


def _web(url: str) -> tuple[str, str, int | None] | None:
    """The scheme, host and port of an http or https URL, else None."""
    try:
        parts = urlsplit(url.strip())
        scheme, hostname, port = parts.scheme.lower(), parts.hostname, parts.port
    except ValueError:
        return None
    if scheme not in LINK_SCHEMES or not hostname:
        return None
    return scheme, hostname, port


def _netloc(hostname: str, port: int | None) -> str:
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{host}:{port}" if port else host


def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"
