"""Story summaries as claims, each tied to a passage copied from the story's sources.

The fast model is asked for structured output: a list of claims, each with the passage
of source text that states it. This code, not the model, then checks that every
passage is a verbatim substring of one source's cleaned text, and ties the claim to
that source's link. A claim whose passage is not found is dropped and listed for the
run report, and a summary call that fails or cannot be read is an error on the story.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from newsletter.issue import MAX_ITEM_CHARS, fill, load_prompt
from newsletter.llm import LLMClient, LLMError
from newsletter.sources import Item

SYSTEM_PROMPT = (
    "You extract the facts of a news story as claims. Use only facts stated in the "
    "sources you are given, and copy each claim's passage exactly from one source. "
    "Reply with JSON only."
)

CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "passage": {"type": "string"},
                },
                "required": ["text", "passage"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "story_claims", "strict": True, "schema": CLAIMS_SCHEMA},
}

_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.DOTALL)


@dataclass(frozen=True)
class Claim:
    id: str
    """`<story id>.c<n>`, numbered in the order the model gave the claims."""
    text: str
    passage: str
    """Found verbatim in the cleaned text of the source at `source_url`."""
    source_url: str


@dataclass(frozen=True)
class DroppedClaim:
    id: str
    story_id: str
    text: str
    passage: str
    reason: str


@dataclass
class StorySummary:
    story_id: str
    claims: list[Claim] = field(default_factory=list)
    dropped: list[DroppedClaim] = field(default_factory=list)
    error: str | None = None
    """Why the story has no claims, when the call failed or its reply could not be read."""


@dataclass
class ClaimsRun:
    summaries: list[StorySummary] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        """Dropped claims and failed stories, for the run report."""
        return {
            "dropped_claims": [asdict(d) for s in self.summaries for d in s.dropped],
            "failed_stories": [
                {"story_id": s.story_id, "error": s.error} for s in self.summaries if s.error
            ],
        }


def summarize_stories(stories: Mapping[str, list[Item]], llm: LLMClient) -> ClaimsRun:
    """One fast-model call per story, keyed by story id, each story given as its sources."""
    return ClaimsRun([summarize_story(sid, items, llm) for sid, items in stories.items()])


def summarize_story(story_id: str, items: list[Item], llm: LLMClient) -> StorySummary:
    """The story's claims. A failed or unreadable call is kept on the summary, not raised."""
    summary = StorySummary(story_id)
    sources = [item for item in items if item.summary.strip()]
    if not sources:
        # Nothing to quote: any claim would be invented.
        return summary
    title = items[0].title
    prompt = fill(load_prompt("claims"), story_title=title, sources=_sources_block(sources))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        reply = llm.complete(
            f"claims-{story_id}", messages, tier="fast", response_format=RESPONSE_FORMAT
        )
        raw_claims = parse_claims(reply)
    except LLMError as error:
        summary.error = str(error)
        return summary
    except ValueError as error:
        summary.error = f"unparsable reply: {error}"
        return summary

    for number, raw in enumerate(raw_claims, start=1):
        claim_id = f"{story_id}.c{number}"
        text, passage, problem = _fields(raw)
        source = None if problem else _source_of(passage, sources)
        if problem is None and source is None:
            problem = "passage not found verbatim in any source of the story"
        if problem:
            summary.dropped.append(DroppedClaim(claim_id, story_id, text, passage, problem))
        else:
            summary.claims.append(Claim(claim_id, text, passage, source.link))
    return summary


def parse_claims(reply: str) -> list[Any]:
    """The `claims` list of a reply; raises `ValueError` when the reply is not that shape.

    Each claim is checked separately by the caller, so one bad claim drops only itself.
    """
    data = read_json(reply)
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError("no list of claims")
    return data["claims"]


def read_json(reply: str) -> Any:
    """The JSON value of a model reply, which may be wrapped in a code fence."""
    text = reply.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"not JSON ({error.msg})") from None


def _fields(raw: Any) -> tuple[str, str, str | None]:
    """The claim's text and stripped passage, and what is wrong with them, if anything."""
    if not isinstance(raw, dict):
        return "", "", "claim is not an object"
    text, passage = raw.get("text"), raw.get("passage")
    if not isinstance(text, str) or not isinstance(passage, str):
        return (
            text if isinstance(text, str) else "",
            passage if isinstance(passage, str) else "",
            "claim text or passage is missing or not a string",
        )
    text, passage = text.strip(), passage.strip()
    if not text:
        return text, passage, "claim text is empty"
    if not passage:
        # An empty passage is a substring of everything, so it proves nothing.
        return text, passage, "passage is empty"
    return text, passage, None


def _source_of(passage: str, sources: list[Item]) -> Item | None:
    """The first source whose cleaned text contains `passage` exactly."""
    return next((item for item in sources if passage in item.summary), None)


def _sources_block(sources: list[Item]) -> str:
    blocks = []
    for number, item in enumerate(sources, start=1):
        blocks.append(
            f"Source {number}: {item.title}\n{item.link}\n\n{item.summary[:MAX_ITEM_CHARS]}"
        )
    return "\n\n---\n\n".join(blocks)
