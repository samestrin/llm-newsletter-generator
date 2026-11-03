"""The voice pass: neutral prose rewritten in the writer's voice, its facts checked unchanged.

Each text is rewritten by the reasoning model from the writer's yourvoice.dev profile,
then checked: a rewrite that drops, changes or adds a link, quoted string, number or
claim id is rejected. A rejected rewrite, a failed or timed-out call, or a profile that
could not be built keeps the neutral text, and each is listed for the run report.
"""

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from newsletter.issue import Issue, fill, load_prompt
from newsletter.llm import LLMClient, LLMError
from newsletter.voice.check import compare_facts
from newsletter.voice.yourvoice import VoiceError, VoiceSettings, read_samples, voice_style

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You rewrite newsletter prose in a writer's own voice. Change the wording only: keep "
    "every fact, link, quoted string, number and claim id exactly as written. Reply with "
    "the rewritten text only."
)


@dataclass(frozen=True)
class KeptNeutral:
    id: str
    reason: str


@dataclass
class VoiceRun:
    texts: dict[str, str] = field(default_factory=dict)
    """Every text by id: the rewrite where it passed the check, else the neutral text."""
    voiced: list[str] = field(default_factory=list)
    kept_neutral: list[KeptNeutral] = field(default_factory=list)
    skipped: str | None = None
    """Why no text was rewritten, when the pass did not run."""
    enabled: bool = False
    """True when `YOURVOICE_API_KEY` is set, so neutral text is a failure, not a choice."""

    @property
    def ok(self) -> bool:
        return not self.kept_neutral and not (self.enabled and self.skipped)

    def report(self) -> dict[str, Any]:
        """Which texts are in the writer's voice and why the others are not."""
        return {
            "enabled": self.enabled,
            "skipped": self.skipped,
            "voiced": list(self.voiced),
            "kept_neutral": [asdict(k) for k in self.kept_neutral],
            "ok": self.ok,
        }


def apply_voice(
    texts: Mapping[str, str],
    llm: LLMClient,
    settings: VoiceSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> VoiceRun:
    """Rewrite each text in the writer's voice; nothing here raises on a failed call."""
    run = VoiceRun(texts=dict(texts), enabled=bool(settings.api_key))
    if not settings.api_key:
        run.skipped = "YOURVOICE_API_KEY is not set"
        return run
    if not texts:
        return run
    samples = read_samples(settings)
    if not samples:
        run.skipped = (
            "no writing samples: set NEWSLETTER_VOICE_SAMPLES to a folder of the writer's "
            ".md or .txt files"
        )
        return run
    if not llm.settings.models.get("reasoning"):
        run.skipped = "no reasoning model: set NEWSLETTER_REASONING_MODEL"
        return run

    try:
        style = voice_style(settings, samples, transport=transport)
    except VoiceError as error:
        run.kept_neutral = [KeptNeutral(i, f"voice profile failed: {error}") for i in texts]
        _log_kept_neutral(run)
        return run

    template = load_prompt("voice")
    for text_id, neutral in texts.items():
        if not neutral.strip():
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fill(template, style=style, text=neutral)},
        ]
        try:
            rewrite = llm.complete(f"voice-{text_id}", messages, tier="reasoning").strip()
        except LLMError as error:
            run.kept_neutral.append(KeptNeutral(text_id, f"rewrite failed: {error}"))
            continue
        problems = compare_facts(neutral, rewrite)
        if problems:
            reason = "rewrite changed facts: " + "; ".join(problems)
            run.kept_neutral.append(KeptNeutral(text_id, reason))
            continue
        run.texts[text_id] = rewrite
        run.voiced.append(text_id)
    _log_kept_neutral(run)
    return run


def voice_issue(
    issue: Issue,
    llm: LLMClient,
    settings: VoiceSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> VoiceRun:
    """Rewrite the issue's summaries in place; the voice and its steps join the run report."""
    summaries = {
        str(number): story.summary
        for number, story in enumerate(issue.stories, start=1)
        if story.summary
    }
    run = apply_voice(summaries, llm, settings, transport=transport)
    for number, story in enumerate(issue.stories, start=1):
        if str(number) in run.texts:
            story.summary = run.texts[str(number)]
    issue.voice = run.report()
    issue.usage = llm.report()
    return run


def _log_kept_neutral(run: VoiceRun) -> None:
    for kept in run.kept_neutral:
        log.warning("voice %s kept neutral: %s", kept.id, kept.reason)
