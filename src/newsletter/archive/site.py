"""A static site of approved issues: an index, and one page per issue.

The site is public, so it is built to show nothing it should not:

- A page's path comes from the issue date and a fixed slug, never from feed text.
- Every piece of feed or model text is HTML-escaped, and pages forbid scripts.
- Only http and https links are rendered; any other link is left out.
- The run report shows failures by feed host and count only: no error body, file
  path, key or model URL. Feeds are credited by title and host, because a feed URL
  can carry a key in its query.

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

from newsletter.issue import Issue, Story

LINK_SCHEMES = ("http", "https")
LOCAL_SOURCE = "a local file"

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


def build_site(issues: Iterable[Issue], site_dir: Path) -> None:
    """Build the site for `issues` into `site_dir`, replacing it only when complete."""
    issues = sorted(issues, key=lambda issue: issue.issue_date, reverse=True)
    dates = Counter(issue.issue_date for issue in issues)
    repeated = sorted(day.isoformat() for day, count in dates.items() if count > 1)
    if repeated:
        raise ArchiveError(f"more than one issue on {', '.join(repeated)}")

    site_dir = Path(site_dir)
    site_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{site_dir.name}-build-", dir=site_dir.parent))
    try:
        _write(work, "index.html", render_index(issues))
        for issue in issues:
            _write(work, issue_path(issue), render_issue(issue))
        work.chmod(0o755)
        _swap(work, site_dir)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def render_index(issues: list[Issue]) -> str:
    lines = ["<h1>Archive</h1>"]
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


def render_issue(issue: Issue) -> str:
    lines = [
        '<p><a href="../../index.html">Archive</a></p>',
        f"<h1>{_text(issue.title)}</h1>",
        f"<p><time>{issue.issue_date.isoformat()}</time></p>",
    ]
    if not issue.stories:
        lines.append("<p>No stories this time: no feed returned an item.</p>")
    for story in issue.stories:
        lines += _story(story)
    lines += _sources(issue.stories)
    lines += _run_report(issue)
    return _PAGE.format(title=_text(issue.title), body="\n".join(lines))


def _story(story: Story) -> list[str]:
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
    problems = Counter(_host(failure.url) for failure in issue.failures)
    problems.update(_host(story.item.feed_url) for story in issue.stories if story.error)
    if problems:
        lines += ["<p>Feeds, entries and summaries left out:</p>", "<ul>"]
        for host, count in sorted(problems.items(), key=lambda pair: (-pair[1], pair[0])):
            lines.append(f"<li>{_text(host)}: {_count(count, 'problem')}</li>")
        lines.append("</ul>")
    else:
        lines.append("<p>Every feed was read and every entry was used.</p>")

    steps = issue.usage.get("steps", [])
    if not steps:
        return [*lines, "<p>No model calls were made.</p>"]
    failed = sum(1 for step in steps if not step.get("ok"))
    calls = f"{_count(len(steps), 'model call')}, {failed} failed"
    cost = f"total cost ${issue.usage.get('total_cost') or 0:.6f}"
    unknown = sum(1 for step in steps if step.get("cost") is None)
    if unknown:
        cost += f", plus a cost unknown for {unknown} of {len(steps)}"
    return [*lines, f"<p>{calls}; {cost}.</p>"]


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
