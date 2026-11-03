"""Approval records, and the gates the archive and the export connectors check.

A human approval is the sha256 of the text a person approved, so any edit voids it.
The daily demo build has no person in the loop, so its approval is a record marked
`automated-demo` with no sha256. It is valid for archive pages only: the pages it
builds are labeled as an automated demo, and `require_export` refuses it, so no
automated issue ever leaves as an email draft.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

AUTOMATED_DEMO = "automated-demo"
HUMAN = "human"
KINDS = (HUMAN, AUTOMATED_DEMO)


class ApprovalError(PermissionError):
    """The issue is not approved for what was asked."""


@dataclass(frozen=True)
class Approval:
    kind: str
    sha256: str | None = None

    @property
    def automated(self) -> bool:
        return self.kind == AUTOMATED_DEMO

    def to_dict(self) -> dict[str, Any]:
        valid_for = "archive" if self.automated else "archive and export"
        return {"kind": self.kind, "sha256": self.sha256, "valid_for": valid_for}


def demo_approval() -> Approval:
    return Approval(kind=AUTOMATED_DEMO)


def human_approval(text: str) -> Approval:
    return Approval(kind=HUMAN, sha256=digest(text))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_archive(approval: Approval | None) -> None:
    """Raise unless `approval` lets the issue onto an archive page."""
    _known(approval)


def require_export(approval: Approval | None, text: str) -> None:
    """Raise unless a person approved exactly `text`. Every export connector calls this."""
    _known(approval)
    if approval.automated:
        raise ApprovalError(
            f"an {AUTOMATED_DEMO} approval is valid for archive pages only, not for export"
        )
    if not approval.sha256:
        raise ApprovalError("the approval has no sha256 of the approved text")
    if approval.sha256 != digest(text):
        raise ApprovalError("the text changed since it was approved")


def _known(approval: Approval | None) -> None:
    if approval is None:
        raise ApprovalError("the issue is not approved")
    if approval.kind not in KINDS:
        raise ApprovalError(f"unknown approval kind {approval.kind!r}")
