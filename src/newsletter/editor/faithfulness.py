"""The faithfulness check: each claim is checked against the passage it was tied to.

One fast-model call per claim answers whether the passage supports the claim. Only a
reply that clearly says yes makes a claim `supported`. A no makes it `unsupported`, and a
call that fails or a reply that cannot be read makes it `unchecked`. Both are flagged,
and every claim stays on its story in order, so a flagged claim is shown as flagged and
never dropped out of sight. Flagged claims and failed checks are listed for the run report.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from newsletter.editor.claims import Claim, ClaimsRun, DroppedClaim, StorySummary, read_json
from newsletter.issue import fill, load_prompt
from newsletter.llm import LLMClient, LLMError

Status = Literal["supported", "unsupported", "unchecked"]

SYSTEM_PROMPT = (
    "You check whether a quoted passage supports a claim. Judge from the passage alone. "
    "Reply with JSON only."
)

CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "reason"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "claim_check", "strict": True, "schema": CHECK_SCHEMA},
}


@dataclass(frozen=True)
class CheckedClaim:
    claim: Claim
    status: Status
    reason: str
    """The model's reason, or why the claim could not be checked."""

    @property
    def id(self) -> str:
        return self.claim.id

    @property
    def flagged(self) -> bool:
        """True unless the check said the passage supports the claim."""
        return self.status != "supported"


@dataclass
class CheckedStory:
    story_id: str
    claims: list[CheckedClaim] = field(default_factory=list)
    """Every claim of the story, flagged or not, in the order of the summary."""
    dropped: list[DroppedClaim] = field(default_factory=list)
    error: str | None = None

    @property
    def supported(self) -> list[CheckedClaim]:
        return [c for c in self.claims if c.status == "supported"]


@dataclass
class CheckRun:
    stories: list[CheckedStory] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        """Flagged claims and failed check calls, for the run report."""
        return {
            "flagged_claims": [
                {
                    "id": c.id,
                    "story_id": s.story_id,
                    "text": c.claim.text,
                    "passage": c.claim.passage,
                    "source_url": c.claim.source_url,
                    "status": c.status,
                    "reason": c.reason,
                }
                for s in self.stories
                for c in s.claims
                if c.flagged
            ],
            "failed_checks": [
                {"id": c.id, "story_id": s.story_id, "error": c.reason}
                for s in self.stories
                for c in s.claims
                if c.status == "unchecked"
            ],
        }


def check_claims(run: ClaimsRun, llm: LLMClient) -> CheckRun:
    """Check every claim of every story; dropped claims and story errors are carried over."""
    return CheckRun([_check_story(summary, llm) for summary in run.summaries])


def check_claim(claim: Claim, llm: LLMClient) -> CheckedClaim:
    """One fast-model check. A failed or unreadable call is `unchecked`, never supported."""
    prompt = fill(load_prompt("faithfulness"), claim=claim.text, passage=claim.passage)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        reply = llm.complete(
            f"check-{claim.id}", messages, tier="fast", response_format=RESPONSE_FORMAT
        )
    except LLMError as error:
        return CheckedClaim(claim, "unchecked", str(error))
    # Parsed apart from the call: a ConfigError is a ValueError too, and must not pass
    # for an unreadable reply.
    try:
        supported, reason = parse_check(reply)
    except ValueError as error:
        return CheckedClaim(claim, "unchecked", f"unparsable reply: {error}")
    return CheckedClaim(claim, "supported" if supported else "unsupported", reason)


def parse_check(reply: str) -> tuple[bool, str]:
    """The verdict and reason of a reply; raises `ValueError` when it is not that shape.

    `supported` must be a JSON boolean: "true", 1 or a missing value is not a yes.
    """
    data = read_json(reply)
    if not isinstance(data, dict):
        raise ValueError("not an object")
    supported, reason = data.get("supported"), data.get("reason")
    if not isinstance(supported, bool):
        raise ValueError("supported is missing or not true or false")
    if not isinstance(reason, str):
        raise ValueError("reason is missing or not a string")
    return supported, reason.strip()


def _check_story(summary: StorySummary, llm: LLMClient) -> CheckedStory:
    return CheckedStory(
        summary.story_id,
        [check_claim(claim, llm) for claim in summary.claims],
        list(summary.dropped),
        summary.error,
    )
