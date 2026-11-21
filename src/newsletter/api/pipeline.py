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
from newsletter.progress import Progress, no_progress, plural
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
    progress: Progress = no_progress,
) -> dict:
    """`progress` hears the start and end of each stage, then `done` as running: the
    caller marks it done once the draft is saved."""
    gathered = gather_stories(
        request.feeds,
        pages=request.pages,
        hackernews=request.hackernews,
        llm=llm,
        topic=request.topic,
        per_feed=request.per_feed,
        fetch=fetch,
        progress=progress,
    )
    stories = gathered.stories

    progress("claims", "running")
    claims = summarize_stories(stories, llm)
    written = sum(len(summary.claims) for summary in claims.summaries)
    progress(
        "claims",
        "done",
        detail=f"{plural(written, 'claim')} from {plural(len(stories), 'story', 'stories')}",
    )

    progress("check", "running")
    checked = check_claims(claims, llm)
    checks = [claim for story in checked.stories for claim in story.claims]
    flagged = sum(1 for claim in checks if claim.flagged)
    progress("check", "done", detail=f"{plural(len(checks), 'claim')} checked, {flagged} flagged")

    progress("editor", "running")
    edition = edit_issue(checked, llm)
    if edition.error:
        progress("editor", "failed")
    else:
        edited = plural(len(edition.stories), "story", "stories")
        progress("editor", "done", detail=f"{edited} in the issue")

    progress("done", "running")
    voiced = voice_edition(edition, llm, voice, transport=voice_transport)
    report = build_report(
        gathered.failures,
        edition,
        stories,
        llm.report(),
        voiced.report(),
        gathered.grouping,
        linked=gathered.linked,
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

    def run(request: RunRequest, progress: Progress = no_progress) -> dict:
        llm = LLMClient(Settings.from_env(env))
        try:
            return run_issue(request, llm, voice=VoiceSettings.from_env(env), progress=progress)
        finally:
            llm.close()

    return run
