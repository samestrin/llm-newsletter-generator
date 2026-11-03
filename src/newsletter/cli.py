"""Command-line entry point for the `newsletter` command.

Exit codes: 0 when the issue (or the archive site) was written and nothing failed; 1
when it was written but a feed, entry or model call failed (the run report says
which), or no feed returned an item; 2 for a usage or settings error, before anything
is fetched.
"""

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import httpx

from newsletter import __version__
from newsletter.archive import build_site, demo_approval
from newsletter.issue import Issue, make_issue, render_markdown
from newsletter.llm import ConfigError, LLMClient, Settings
from newsletter.sources import FeedCache, RssSource
from newsletter.sources.http import Fetch, http_get
from newsletter.voice import VoiceSettings, voice_issue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsletter",
        description="Generate text-only newsletters from RSS feeds using LLMs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    drafting = argparse.ArgumentParser(add_help=False)
    drafting.add_argument("feeds", nargs="+", metavar="FEED_URL", help="RSS or Atom feed URL")
    drafting.add_argument("--title", default="Newsletter", help="issue title")
    drafting.add_argument("--topic", default="news", help="topic given to the model")
    drafting.add_argument("--cache-dir", type=Path, help="keep fetched feeds here between runs")
    drafting.add_argument(
        "--date", type=date.fromisoformat, default=None, help="issue date, YYYY-MM-DD"
    )
    drafting.add_argument(
        "--per-feed",
        type=_positive,
        metavar="N",
        help="summarize only the first N items of each feed (each item is one model call)",
    )

    issue = commands.add_parser(
        "issue",
        parents=[drafting],
        help="draft an issue from feeds: one summary and one link per item",
        description=(
            "Read RSS or Atom feeds, summarize each item with the fast model "
            "(NEWSLETTER_FAST_MODEL) and write the issue as Markdown, with a link per item "
            "and a run report of anything that failed. When YOURVOICE_API_KEY is set, the "
            "summaries are rewritten in the writer's voice (see the README)."
        ),
    )
    issue.add_argument("-o", "--output", type=Path, help="write here instead of stdout")

    archive = commands.add_parser(
        "archive",
        parents=[drafting],
        help="draft an issue from feeds and build the static archive site (demo mode)",
        description=(
            "Draft an issue as `newsletter issue` does and build it into a static site. "
            "Only demo mode exists so far: the issue is approved by an automated-demo "
            "record, valid for archive pages only, and every page is labeled as an "
            "automated demo. Nothing is published or exported. The writer's voice applies "
            "as for `newsletter issue`."
        ),
    )
    archive.add_argument("--site", type=Path, required=True, help="build the site here")
    archive.add_argument(
        "--demo",
        action="store_true",
        help="approve with an automated-demo record and label every page; required",
    )

    serve = commands.add_parser(
        "serve",
        help="run the API that drafts issues for review",
        description=(
            "Serve the HTTP API: run an issue as a background job, list, edit and approve "
            "drafts. Needs NEWSLETTER_API_TOKEN; binds 127.0.0.1:8000 unless told otherwise "
            "(see the README)."
        ),
    )
    serve.add_argument("--host", help="bind address (NEWSLETTER_API_HOST, default 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (NEWSLETTER_API_PORT, default 8000)")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    fetch: Fetch = http_get,
    llm_transport: httpx.BaseTransport | None = None,
    voice_transport: httpx.BaseTransport | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "archive":
        return _archive(args, env, fetch, llm_transport, voice_transport)
    if args.command == "serve":
        return _serve(args, env)
    return _issue(args, env, fetch, llm_transport, voice_transport)


def _serve(args, env) -> int:
    import uvicorn

    from newsletter.api import ApiSettings, create_app

    env = dict(os.environ if env is None else env)
    if args.host:
        env["NEWSLETTER_API_HOST"] = args.host
    if args.port is not None:
        env["NEWSLETTER_API_PORT"] = str(args.port)
    try:
        settings = ApiSettings.from_env(env)
    except ConfigError as error:
        print(f"newsletter: {error}", file=sys.stderr)
        return 2
    if not settings.token:
        print("newsletter: set NEWSLETTER_API_TOKEN before serving the API", file=sys.stderr)
        return 2
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


def _issue(args, env, fetch, llm_transport, voice_transport) -> int:
    issue = _draft(args, env, fetch, llm_transport, voice_transport)
    if issue is None:
        return 2

    markdown = render_markdown(issue)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    return _summary(issue, args.output)


def _archive(args, env, fetch, llm_transport, voice_transport) -> int:
    if not args.demo:
        print(
            "newsletter: the archive needs an approval, and only --demo (an automated-demo "
            "approval, labeled on every page) is available",
            file=sys.stderr,
        )
        return 2
    issue = _draft(args, env, fetch, llm_transport, voice_transport)
    if issue is None:
        return 2
    build_site([issue], args.site, approval=demo_approval())
    return _summary(issue, args.site)


def _draft(args, env, fetch, llm_transport, voice_transport) -> Issue | None:
    """The issue drafted from `args.feeds` (in the writer's voice when it is set up), or
    None after printing a settings error."""
    try:
        settings = Settings.from_env(env)
        voice = VoiceSettings.from_env(env)
    except ConfigError as error:
        print(f"newsletter: {error}", file=sys.stderr)
        return None
    if not settings.models.get("fast"):
        print("newsletter: no fast model: set NEWSLETTER_FAST_MODEL", file=sys.stderr)
        return None

    cache = FeedCache(args.cache_dir) if args.cache_dir else None
    llm = LLMClient(settings, transport=llm_transport)
    try:
        issue = make_issue(
            args.feeds,
            title=args.title,
            topic=args.topic,
            source=RssSource(cache=cache, fetch=fetch),
            llm=llm,
            issue_date=args.date or date.today(),
            per_feed=args.per_feed,
        )
        voice_issue(issue, llm, voice, transport=voice_transport)
    finally:
        llm.close()
    return issue


def _summary(issue: Issue, written: Path | None) -> int:
    failed = sum(1 for story in issue.stories if story.error)
    print(
        f"newsletter: {_count(len(issue.stories), 'story', 'stories')}, "
        f"{_count(len(issue.failures), 'feed problem')}, "
        f"{_count(failed, 'failed summary', 'failed summaries')}"
        + (f"; wrote {written}" if written else ""),
        file=sys.stderr,
    )
    return 0 if issue.ok else 1


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"
