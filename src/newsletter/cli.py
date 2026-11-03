"""Command-line entry point for the `newsletter` command.

Exit codes: 0 when the issue was written and nothing failed; 1 when the issue was
written but a feed, entry or model call failed (the run report says which), or no
feed returned an item; 2 for a usage or settings error, before anything is fetched.
"""

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import httpx

from newsletter import __version__
from newsletter.issue import make_issue, render_markdown
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

    issue = commands.add_parser(
        "issue",
        help="draft an issue from feeds: one summary and one link per item",
        description=(
            "Read RSS or Atom feeds, summarize each item with the fast model "
            "(NEWSLETTER_FAST_MODEL) and write the issue as Markdown, with a link per item "
            "and a run report of anything that failed. When YOURVOICE_API_KEY is set, the "
            "summaries are rewritten in the writer's voice (see the README)."
        ),
    )
    issue.add_argument("feeds", nargs="+", metavar="FEED_URL", help="RSS or Atom feed URL")
    issue.add_argument("--title", default="Newsletter", help="issue title")
    issue.add_argument("--topic", default="news", help="topic given to the model")
    issue.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    issue.add_argument("--cache-dir", type=Path, help="keep fetched feeds here between runs")
    issue.add_argument(
        "--date", type=date.fromisoformat, default=None, help="issue date, YYYY-MM-DD"
    )
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
    return _issue(args, env, fetch, llm_transport, voice_transport)


def _issue(args, env, fetch, llm_transport, voice_transport) -> int:
    try:
        settings = Settings.from_env(env)
        voice = VoiceSettings.from_env(env)
    except ConfigError as error:
        print(f"newsletter: {error}", file=sys.stderr)
        return 2
    if not settings.models.get("fast"):
        print("newsletter: no fast model: set NEWSLETTER_FAST_MODEL", file=sys.stderr)
        return 2

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
        )
        voice_issue(issue, llm, voice, transport=voice_transport)
    finally:
        llm.close()

    markdown = render_markdown(issue)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)

    failed = sum(1 for story in issue.stories if story.error)
    print(
        f"newsletter: {_count(len(issue.stories), 'story', 'stories')}, "
        f"{_count(len(issue.failures), 'feed problem')}, "
        f"{_count(failed, 'failed summary', 'failed summaries')}"
        + (f"; wrote {args.output}" if args.output else ""),
        file=sys.stderr,
    )
    return 0 if issue.ok else 1


def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"
