"""The editor pass: the reasoning model edits the issue from supported claims only.

The model is given each story's supported claims with their ids, and nothing else, and
asked to pick the lead story, order the stories, name 2 or 3 themes, and write the intro
and a "why it matters" line per story. Every piece of prose cites the claim ids it rests
on. This code, not the model, then checks that each cited id is known and supported; a
reply that cites anything else, or is not the asked shape, is rejected and the model is
asked once more with the reasons. A second rejection, or a failed call, is an error on
the edition: no prose from the model is used, and the stories stay listed in their
original order, flagged claims and all.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from newsletter.editor.claims import read_json
from newsletter.editor.faithfulness import CheckedStory, CheckRun
from newsletter.issue import fill, load_prompt
from newsletter.llm import LLMClient, LLMError

SYSTEM_PROMPT = (
    "You are the editor of a newsletter. Write only from the claims you are given, and "
    "cite the id of every claim each piece of your writing rests on. Reply with JSON only."
)

MIN_THEMES, MAX_THEMES = 2, 3

_IDS = {"type": "array", "items": {"type": "string"}}

EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lead": {"type": "string"},
        "order": {"type": "array", "items": {"type": "string"}},
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "claim_ids": _IDS},
                "required": ["name", "claim_ids"],
                "additionalProperties": False,
            },
        },
        "intro": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "claim_ids": _IDS},
            "required": ["text", "claim_ids"],
            "additionalProperties": False,
        },
        "why_it_matters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "story_id": {"type": "string"},
                    "text": {"type": "string"},
                    "claim_ids": _IDS,
                },
                "required": ["story_id", "text", "claim_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lead", "order", "themes", "intro", "why_it_matters"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "issue_edit", "strict": True, "schema": EDIT_SCHEMA},
}


class InvalidEdit(ValueError):
    """The editor's reply cannot be used; `problems` says why, one reason per item."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Prose:
    text: str
    claim_ids: list[str]
    """Each one a supported claim."""


@dataclass(frozen=True)
class Theme:
    name: str
    claim_ids: list[str]


@dataclass(frozen=True)
class EditDraft:
    """A reply that passed the validator."""

    lead: str
    order: list[str]
    themes: list[Theme]
    intro: Prose
    why_it_matters: dict[str, Prose]


@dataclass
class Edition:
    stories: list[CheckedStory]
    """Every story, flagged claims included: in the editor's order, then those it could
    not write about; in their original order when the editor pass failed."""
    lead: str | None = None
    themes: list[Theme] = field(default_factory=list)
    intro: Prose | None = None
    why_it_matters: dict[str, Prose] = field(default_factory=dict)
    rejections: list[str] = field(default_factory=list)
    """Why each rejected reply was rejected, kept even when the retry was good."""
    error: str | None = None
    """Why there is no edit, when the call failed or both replies were rejected."""

    def report(self) -> dict[str, Any]:
        """The editor's failure and rejected replies, for the run report."""
        return {"editor_error": self.error, "editor_rejections": list(self.rejections)}


def edit_issue(run: CheckRun, llm: LLMClient) -> Edition:
    """One reasoning call, and one retry if its reply is rejected. Failures are kept on
    the edition, not raised."""
    edition = Edition(list(run.stories))
    editable = [story for story in run.stories if story.supported]
    if not editable:
        edition.error = "no supported claims to edit from"
        return edition

    prompt = fill(load_prompt("editor"), stories=_stories_block(editable))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for step in ("edit", "edit-retry"):
        try:
            reply = llm.complete(step, messages, tier="reasoning", response_format=RESPONSE_FORMAT)
        except LLMError as error:
            edition.error = str(error)
            return edition
        # Parsed apart from the call: a ConfigError is a ValueError too.
        try:
            draft = parse_edit(reply, run.stories)
        except InvalidEdit as invalid:
            edition.rejections.append(f"{step}: {invalid}")
            retry = fill(load_prompt("editor_retry"), problems=_bullets(invalid.problems))
            messages = [
                *messages,
                {"role": "assistant", "content": reply},
                {"role": "user", "content": retry},
            ]
            continue
        return _apply(edition, draft)
    edition.error = f"editor output rejected twice; last: {edition.rejections[-1]}"
    return edition


def parse_edit(reply: str, stories: list[CheckedStory]) -> EditDraft:
    """The edit in a reply, checked against `stories`; raises `InvalidEdit` otherwise.

    Only stories with a supported claim can be ordered, and every piece of prose must
    cite at least one claim id, each known and supported.
    """
    try:
        data = read_json(reply)
    except ValueError as error:
        raise InvalidEdit([str(error)]) from None
    if not isinstance(data, dict):
        raise InvalidEdit(["reply is not an object"])
    missing = [key for key in EDIT_SCHEMA["required"] if key not in data]
    if missing:
        raise InvalidEdit([f"reply has no {', '.join(missing)}"])

    lead, order = data["lead"], data["order"]
    if not isinstance(lead, str):
        raise InvalidEdit(["lead is not a story id"])
    if not _strings(order):
        raise InvalidEdit(["order is not a list of story ids"])
    if not isinstance(data["themes"], list) or not isinstance(data["why_it_matters"], list):
        raise InvalidEdit(["themes and why it matters must be lists"])

    statuses = {c.id: (s.story_id, c.status) for s in stories for c in s.claims}
    editable = [s.story_id for s in stories if s.supported]
    problems: list[str] = []

    def prose(raw: Any, label: str, *, text_key: str = "text") -> Prose:
        if not isinstance(raw, dict) or not isinstance(raw.get(text_key), str):
            raise InvalidEdit([f"{label} is not text with claim ids"])
        text, ids = raw[text_key].strip(), raw.get("claim_ids")
        if not _strings(ids):
            problems.append(f"{label} claim ids are not strings")
            return Prose(text, [])
        if not text:
            problems.append(f"{label} is empty")
        if not ids:
            problems.append(f"{label} cites no claim")
        for claim_id in ids:
            if claim_id not in statuses:
                problems.append(f"unknown claim id {claim_id!r}")
            elif statuses[claim_id][1] != "supported":
                problems.append(f"claim {claim_id!r} is {statuses[claim_id][1]}")
        return Prose(text, list(ids))

    for story_id, count in Counter(order).items():
        if story_id not in editable:
            problems.append(f"order has unknown story {story_id!r}")
        elif count > 1:
            problems.append(f"order lists {story_id!r} twice")
    problems.extend(f"order leaves out {s!r}" for s in editable if s not in order)
    if not order or order[0] != lead:
        problems.append(f"lead {lead!r} is not first in the order")

    if not MIN_THEMES <= len(data["themes"]) <= MAX_THEMES:
        problems.append(
            f"{MIN_THEMES} or {MAX_THEMES} themes are needed, got {len(data['themes'])}"
        )
    themes = []
    for number, raw in enumerate(data["themes"], start=1):
        named = prose(raw, f"theme {number}", text_key="name")
        themes.append(Theme(named.text, named.claim_ids))

    intro = prose(data["intro"], "intro")

    why: dict[str, Prose] = {}
    for raw in data["why_it_matters"]:
        story_id = raw.get("story_id") if isinstance(raw, dict) else None
        if not isinstance(story_id, str):
            raise InvalidEdit(["why it matters has an entry with no story id"])
        label = f"why it matters for {story_id!r}"
        if story_id not in editable:
            problems.append(f"why it matters for unknown story {story_id!r}")
        elif story_id in why:
            problems.append(f"{label} given twice")
        line = prose(raw, label)
        for claim_id in line.claim_ids:
            owner = statuses.get(claim_id, (story_id,))[0]
            if owner != story_id:
                problems.append(f"{label} cites {claim_id!r} of another story")
        why.setdefault(story_id, line)
    problems.extend(f"no why it matters for {s!r}" for s in editable if s not in why)

    if problems:
        raise InvalidEdit(list(dict.fromkeys(problems)))
    return EditDraft(lead, list(order), themes, intro, why)


def _apply(edition: Edition, draft: EditDraft) -> Edition:
    by_id = {story.story_id: story for story in edition.stories}
    rest = [story for story in edition.stories if story.story_id not in draft.order]
    edition.stories = [by_id[story_id] for story_id in draft.order] + rest
    edition.lead = draft.lead
    edition.themes = draft.themes
    edition.intro = draft.intro
    edition.why_it_matters = draft.why_it_matters
    return edition


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _stories_block(stories: list[CheckedStory]) -> str:
    blocks = []
    for story in stories:
        claims = "\n".join(f"- {c.id}: {c.claim.text}" for c in story.supported)
        blocks.append(f"Story {story.story_id}\n{claims}")
    return "\n\n".join(blocks)
