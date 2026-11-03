"""One run of an issue: read the feeds, the editor pass, then the voice pass, then render.

The result is what a draft holds: the rendered Markdown, the flagged claims and the run
report. A failed feed, model call or rewrite is kept in the report, never raised.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date

import httpx

from newsletter.editor import Edition, check_claims, edit_issue, summarize_stories
from newsletter.llm import LLMClient, Settings
from newsletter.render import Newsletter, build_report, render_markdown
from newsletter.sources import RssSource
from newsletter.voice import VoiceRun, VoiceSettings, apply_voice


@dataclass(frozen=True)
class RunRequest:
    feeds: list[str]
    title: str = "Newsletter"
    topic: str = "news"
    issue_date: date | None = None


def run_issue(
    request: RunRequest,
    llm: LLMClient,
    *,
    source: RssSource,
    voice: VoiceSettings,
    voice_transport: httpx.BaseTransport | None = None,
) -> dict:
    collected = source.collect(request.feeds)
    stories = {f"s{n}": [item] for n, item in enumerate(collected.items, start=1)}
    claims = summarize_stories(stories, llm)
    checked = check_claims(claims, llm)
    edition = edit_issue(checked, llm)
    voiced = voice_edition(edition, llm, voice, transport=voice_transport)
    report = build_report(collected.failures, edition, stories, llm.report(), voiced.report())
    issue_date = request.issue_date or date.today()
    text = render_markdown(Newsletter(request.title, issue_date, edition, stories, report))
    return {
        "title": request.title,
        "issue_date": issue_date.isoformat(),
        "text": text,
        "flags": [asdict(flag) for flag in report.flagged_claims],
        "report": report.to_dict(),
    }


def voice_edition(
    edition: Edition,
    llm: LLMClient,
    settings: VoiceSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> VoiceRun:
    """Rewrite the editor's prose in place: the intro and each why it matters. Claim ids
    stay with their prose; the claims themselves are never rewritten."""
    texts = {f"why-{story_id}": p.text for story_id, p in edition.why_it_matters.items()}
    if edition.intro is not None:
        texts = {"intro": edition.intro.text, **texts}
    run = apply_voice(texts, llm, settings, transport=transport)
    if edition.intro is not None:
        edition.intro = replace(edition.intro, text=run.texts["intro"])
    for story_id, prose in edition.why_it_matters.items():
        edition.why_it_matters[story_id] = replace(prose, text=run.texts[f"why-{story_id}"])
    return run


def default_runner(env: Mapping[str, str]):
    """The runner the API uses: settings read from `env` at each run, a fresh client each."""

    def run(request: RunRequest) -> dict:
        llm = LLMClient(Settings.from_env(env))
        try:
            return run_issue(request, llm, source=RssSource(), voice=VoiceSettings.from_env(env))
        finally:
            llm.close()

    return run
