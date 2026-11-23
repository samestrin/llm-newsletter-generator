"""One run of an issue: read the sources and group their stories, the editor pass, then
the voice pass, then render.

The result is what a draft holds: the rendered Markdown, the flagged claims and the run
report. A failed source, embedding call, model call or rewrite is kept in the report, and
every item left out of grouping is listed, never raised.

A run's model settings are the server's (`NEWSLETTER_*`), or, when the request named a
provider, that provider's with the visitor's key, or the demo gateway's with a demo code
(`run_settings`). The key or code lives only on the run's `RunRequest` and its
`LLMClient`; `Jobs` redacts it from everything the run stores, returns or logs.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import date

import httpx

from newsletter.api.credentials import DEMO, Credentials
from newsletter.config import HackerNewsConfig
from newsletter.editor import check_claims, edit_issue, summarize_stories
from newsletter.issue import gather_stories
from newsletter.llm import ConfigError, LLMClient, Settings
from newsletter.progress import Progress, no_progress, plural
from newsletter.render import Newsletter, build_report, render_markdown
from newsletter.sources.http import Fetch, http_get
from newsletter.sources.rss import FeedFailure
from newsletter.voice import VoiceSettings, voice_edition

NO_EMBEDDING = "{label} has no embedding model: stories are not grouped by meaning"
"""The run's note when its provider has no embedding model, in the grouping step and
the report."""
DEMO_BASE_URL = "NEWSLETTER_DEMO_BASE_URL"


@dataclass(frozen=True)
class RunRequest:
    feeds: list[str]
    title: str = "Newsletter"
    topic: str = "news"
    issue_date: date | None = None
    pages: list[str] = field(default_factory=list)
    hackernews: HackerNewsConfig | None = None
    per_feed: int | None = None
    failed_feeds: list[FeedFailure] = field(default_factory=list)
    """Feeds that failed the check before the run: not read, listed in the feeds step."""
    credentials: Credentials | None = field(default=None, repr=False)
    """The provider and the visitor's key or demo code; `None` for the server's own key.
    Never stored: the run record is built from the fields above it."""


def run_settings(env: Mapping[str, str], credentials: Credentials | None) -> Settings:
    """The model settings of one run. Without credentials, the server's. With a visitor
    key, the provider's base URL and models and that key. With a demo code, the demo
    gateway (`NEWSLETTER_DEMO_BASE_URL`) and its models (`NEWSLETTER_DEMO_<TIER>_MODEL`,
    else the demo provider's names), with the code as the bearer. Timeouts and retries
    stay the server's; the server's prices are for its own models, so none are kept."""
    server = Settings.from_env(env)
    if credentials is None:
        return server
    provider = credentials.provider
    models = {
        "fast": provider.fast_model,
        "reasoning": provider.reasoning_model,
        "embedding": provider.embedding_model,
    }
    if provider.id == DEMO:
        base_url = (env.get(DEMO_BASE_URL) or "").strip().rstrip("/")
        if not base_url:
            raise ConfigError(f"demo codes are off: set {DEMO_BASE_URL}")
        for tier in models:
            models[tier] = env.get(f"NEWSLETTER_DEMO_{tier.upper()}_MODEL") or models[tier]
    else:
        base_url = provider.base_url
    return replace(
        server,
        base_url=base_url,
        api_key=credentials.secret,
        models={tier: model for tier, model in models.items() if model},
        prices={},
    )


def run_issue(
    request: RunRequest,
    llm: LLMClient,
    *,
    voice: VoiceSettings,
    fetch: Fetch = http_get,
    voice_transport: httpx.BaseTransport | None = None,
    progress: Progress = no_progress,
) -> dict:
    """`progress` hears the start and end of each stage, with the feeds, articles, claims
    and checks counted `done` of `total` as they go, then `done` as running: the caller
    marks it done once the draft is saved."""
    provider = request.credentials.provider if request.credentials else None
    ungrouped = None
    if provider is not None and provider.embedding_model is None:
        ungrouped = NO_EMBEDDING.format(label=provider.label)
    gathered = gather_stories(
        request.feeds,
        pages=request.pages,
        hackernews=request.hackernews,
        llm=llm,
        topic=request.topic,
        per_feed=request.per_feed,
        fetch=fetch,
        progress=progress,
        ungrouped=ungrouped,
        failed=request.failed_feeds,
    )
    stories = gathered.stories

    def on_story(number: int, total: int) -> None:
        progress(
            "claims", "running", detail=f"story {number} of {total}", done=number - 1, total=total
        )

    progress("claims", "running", done=0, total=len(stories))
    claims = summarize_stories(stories, llm, on_story=on_story)
    written = sum(len(summary.claims) for summary in claims.summaries)
    progress(
        "claims",
        "done",
        detail=f"{plural(written, 'claim')} from {plural(len(stories), 'story', 'stories')}",
        done=len(stories),
        total=len(stories),
    )

    def on_check(done: int, total: int) -> None:
        progress("check", "running", detail=f"{done} of {total}", done=done, total=total)

    checked = check_claims(claims, llm, on_check=on_check)
    checks = [claim for story in checked.stories for claim in story.claims]
    flagged = sum(1 for claim in checks if claim.flagged)
    progress(
        "check",
        "done",
        detail=f"{plural(len(checks), 'claim')} checked, {flagged} flagged",
        done=len(checks),
        total=len(checks),
    )

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


def default_runner(
    env: Mapping[str, str],
    *,
    transport: httpx.BaseTransport | None = None,
    fetch: Fetch = http_get,
):
    """The runner the API uses: settings read from `env` and the request's credentials at
    each run, a fresh client each. `transport` and `fetch` are for tests."""

    def run(request: RunRequest, progress: Progress = no_progress) -> dict:
        llm = LLMClient(run_settings(env, request.credentials), transport=transport)
        try:
            return run_issue(
                request,
                llm,
                voice=VoiceSettings.from_env(env),
                fetch=fetch,
                progress=progress,
            )
        finally:
            llm.close()

    return run
