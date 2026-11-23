"""The run report: everything that failed, was left out or is in doubt, and what it cost.

It is built from the sources' failures, the link-only items whose article was read or
skipped, the grouping step's error or note and the items it
left out, the edited issue (dropped claims, flagged claims,
stories with nothing to show, the editor's own failure), the voice pass's
`VoiceRun.report()` when there was one, and `LLMClient.report()`, whose steps and costs
are used as they are, never recomputed. An unknown cost stays unknown.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from newsletter.editor import DroppedClaim, Edition
from newsletter.editor.explain import explain
from newsletter.editor.faithfulness import Status
from newsletter.grouping import Grouping, LeftOut
from newsletter.sources import FeedFailure, Item, LinkedArticle


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
    """The check's own reason: the model's, or why the check failed."""
    source_url: str
    passage: str = ""
    """The source text the claim was tied to (`Claim.passage`)."""
    explanation: str = ""
    """Why it is flagged, in plain words built in code (`editor.explain`)."""


@dataclass
class RunReport:
    failed_sources: list[FeedFailure] = field(default_factory=list)
    """Sources (feeds, pages, Hacker News) that could not be read and entries left out."""
    linked_articles: list[LinkedArticle] = field(default_factory=list)
    """Link-only items whose linked article was read for their text, or skipped and why."""
    left_out: list[LeftOut] = field(default_factory=list)
    """Items the grouping step dropped or could not group with the others."""
    grouping_error: str | None = None
    """Why the grouping step failed, when it did: then every item is its own story."""
    grouping_note: str | None = None
    """Why stories were not grouped by meaning when nothing failed: the provider has no
    embedding model. Every item is its own story; this alone does not make `ok` false."""
    dropped_stories: list[DroppedStory] = field(default_factory=list)
    dropped_claims: list[DroppedClaim] = field(default_factory=list)
    flagged_claims: list[FlaggedClaim] = field(default_factory=list)
    editor_error: str | None = None
    """The editor's raw error, for the report details only."""
    editor_message: str | None = None
    """The editor's failure in one plain sentence, built in code: what a reader sees."""
    editor_rejections: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    """`LLMClient.report()["steps"]`: step, tier, model, tokens, cost (None if unknown)."""
    total_cost: float = 0.0
    """The sum of the known costs only."""
    cost_complete: bool = True
    failed_steps: list[str] = field(default_factory=list)
    voice: dict[str, Any] = field(default_factory=dict)
    """`VoiceRun.report()`: which texts are in the writer's voice and why the others are
    not. Empty when no voice pass ran."""

    @property
    def unknown_costs(self) -> int:
        return sum(1 for step in self.steps if step.get("cost") is None)

    @property
    def ok(self) -> bool:
        """True only when nothing failed, was left out or is flagged."""
        return not (
            self.failed_sources
            or any(article.status == "skipped" for article in self.linked_articles)
            or self.left_out
            or self.grouping_error
            or self.dropped_stories
            or self.dropped_claims
            or self.flagged_claims
            or self.editor_error
            or self.failed_steps
            or not self.voice.get("ok", True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def build_report(
    failures: list[FeedFailure],
    edition: Edition,
    stories: Mapping[str, list[Item]],
    usage: Mapping[str, Any],
    voice: Mapping[str, Any] | None = None,
    grouping: Grouping | None = None,
    linked: list[LinkedArticle] | None = None,
) -> RunReport:
    """The report of one run. `stories` maps each story id to its source items, `usage`
    is `LLMClient.report()`, `voice` is `VoiceRun.report()` and `grouping` is the
    `group_items` result the stories came from, and `linked` the link-only items whose
    article was read or skipped."""
    report = RunReport(
        failed_sources=list(failures),
        linked_articles=list(linked or []),
        left_out=list(grouping.left_out) if grouping else [],
        grouping_error=grouping.error if grouping else None,
        grouping_note=grouping.note if grouping else None,
        editor_error=edition.error,
        editor_message=edition.message,
        editor_rejections=list(edition.rejections),
        steps=[dict(step) for step in usage.get("steps", [])],
        total_cost=usage.get("total_cost", 0.0),
        cost_complete=usage.get("cost_complete", True),
        failed_steps=list(usage.get("failed_steps", [])),
        voice=dict(voice or {}),
    )
    for story in edition.stories:
        report.dropped_claims += story.dropped
        report.flagged_claims += [
            FlaggedClaim(
                c.id,
                story.story_id,
                c.claim.text,
                c.status,
                c.reason,
                c.claim.source_url,
                c.claim.passage,
                explain(c.claim.text, c.claim.passage, c.status, c.reason),
            )
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
