"""One run of an issue: read the sources and group their stories, the editor pass, then
the voice pass, then render.

The result is what a draft holds: the rendered Markdown, the flagged claims and the run
report. A failed source, embedding call, model call or rewrite is kept in the report, and
every item left out of grouping is listed, never raised.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date

import httpx

from newsletter.config import HackerNewsConfig
from newsletter.editor import check_claims, edit_issue, summarize_stories
from newsletter.issue import gather_stories
from newsletter.llm import LLMClient, Settings
from newsletter.render import Newsletter, build_report, render_markdown
from newsletter.sources.http import Fetch, http_get
from newsletter.voice import VoiceSettings, voice_edition


@dataclass(frozen=True)
class RunRequest:
    feeds: list[str]
    title: str = "Newsletter"
    topic: str = "news"
    issue_date: date | None = None
    pages: list[str] = field(default_factory=list)
    hackernews: HackerNewsConfig | None = None
    per_feed: int | None = None


def run_issue(
    request: RunRequest,
    llm: LLMClient,
    *,
    voice: VoiceSettings,
    fetch: Fetch = http_get,
    voice_transport: httpx.BaseTransport | None = None,
) -> dict:
    gathered = gather_stories(
        request.feeds,
        pages=request.pages,
        hackernews=request.hackernews,
        llm=llm,
        topic=request.topic,
        per_feed=request.per_feed,
        fetch=fetch,
    )
    stories = gathered.stories
    claims = summarize_stories(stories, llm)
    checked = check_claims(claims, llm)
    edition = edit_issue(checked, llm)
    voiced = voice_edition(edition, llm, voice, transport=voice_transport)
    report = build_report(
        gathered.failures, edition, stories, llm.report(), voiced.report(), gathered.grouping
    )
    issue_date = request.issue_date or date.today()
    text = render_markdown(Newsletter(request.title, issue_date, edition, stories, report))
    return {
        "title": request.title,
        "issue_date": issue_date.isoformat(),
        "text": text,
        "flags": [asdict(flag) for flag in report.flagged_claims],
        "report": report.to_dict(),
    }


def default_runner(env: Mapping[str, str]):
    """The runner the API uses: settings read from `env` at each run, a fresh client each."""

    def run(request: RunRequest) -> dict:
        llm = LLMClient(Settings.from_env(env))
        try:
            return run_issue(request, llm, voice=VoiceSettings.from_env(env))
        finally:
            llm.close()

    return run
