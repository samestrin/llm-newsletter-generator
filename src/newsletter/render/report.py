"""The run report: everything that failed, was left out or is in doubt, and what it cost.

It is built from the feeds' failures, the edited issue (dropped claims, flagged claims,
stories with nothing to show, the editor's own failure) and `LLMClient.report()`, whose
steps and costs are used as they are, never recomputed. An unknown cost stays unknown.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from newsletter.editor import DroppedClaim, Edition
from newsletter.editor.faithfulness import Status
from newsletter.sources import FeedFailure, Item


@dataclass(frozen=True)
class DroppedStory:
    """A story left out of the issue because it has no claim to show."""

    story_id: str
    title: str
    link: str
    reason: str


@dataclass(frozen=True)
class FlaggedClaim:
    """A claim shown in the issue, marked: its source does not support it, or it was not
    checked."""

    id: str
    story_id: str
    text: str
    status: Status
    reason: str
    source_url: str


@dataclass
class RunReport:
    failed_sources: list[FeedFailure] = field(default_factory=list)
    """Feeds that could not be read and feed entries left out."""
    dropped_stories: list[DroppedStory] = field(default_factory=list)
    dropped_claims: list[DroppedClaim] = field(default_factory=list)
    flagged_claims: list[FlaggedClaim] = field(default_factory=list)
    editor_error: str | None = None
    editor_rejections: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    """`LLMClient.report()["steps"]`: step, tier, model, tokens, cost (None if unknown)."""
    total_cost: float = 0.0
    """The sum of the known costs only."""
    cost_complete: bool = True
    failed_steps: list[str] = field(default_factory=list)

    @property
    def unknown_costs(self) -> int:
        return sum(1 for step in self.steps if step.get("cost") is None)

    @property
    def ok(self) -> bool:
        """True only when nothing failed, was left out or is flagged."""
        return not (
            self.failed_sources
            or self.dropped_stories
            or self.dropped_claims
            or self.flagged_claims
            or self.editor_error
            or self.failed_steps
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def build_report(
    failures: list[FeedFailure],
    edition: Edition,
    stories: Mapping[str, list[Item]],
    usage: Mapping[str, Any],
) -> RunReport:
    """The report of one run. `stories` maps each story id to its source items, and
    `usage` is `LLMClient.report()`."""
    report = RunReport(
        failed_sources=list(failures),
        editor_error=edition.error,
        editor_rejections=list(edition.rejections),
        steps=[dict(step) for step in usage.get("steps", [])],
        total_cost=usage.get("total_cost", 0.0),
        cost_complete=usage.get("cost_complete", True),
        failed_steps=list(usage.get("failed_steps", [])),
    )
    for story in edition.stories:
        report.dropped_claims += story.dropped
        report.flagged_claims += [
            FlaggedClaim(c.id, story.story_id, c.claim.text, c.status, c.reason, c.claim.source_url)
            for c in story.claims
            if c.flagged
        ]
        if not story.claims:
            if story.error:
                reason = story.error
            elif story.dropped:
                reason = f"every claim was dropped ({len(story.dropped)})"
            else:
                reason = "no claims: the sources gave no text"
            items = stories.get(story.story_id) or []
            title, link = (items[0].title, items[0].link) if items else (story.story_id, "")
            report.dropped_stories.append(DroppedStory(story.story_id, title, link, reason))
    return report
