"""Export an approved issue to every configured platform, and report what happened.

The approval is checked before any request is made: only a person's approval of exactly
this text lets it leave (`newsletter.archive.require_export`). Each connector then makes
its one create call. A failure is reported with its reason and never counted as an
export, and the others still run; there is no retry, since a repeated create call
could leave a second draft. Every key is redacted from what is logged and reported.
"""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from newsletter.archive.approval import HUMAN, Approval, require_export
from newsletter.export.connectors import Connector, ExportError
from newsletter.llm import ERROR_BODY_LIMIT

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportResult:
    connector: str
    ok: bool
    draft_id: str | None = None
    error: str | None = None


@dataclass
class ExportReport:
    results: list[ExportResult] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    """Connectors left out because their platform has no draft-only endpoint."""

    @property
    def exported(self) -> list[ExportResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[ExportResult]:
        return [r for r in self.results if not r.ok]

    @property
    def ok(self) -> bool:
        """True only when at least one draft was made and no connector failed."""
        return bool(self.results) and not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "results": [vars(r) for r in self.results],
            "dropped": list(self.dropped),
        }

    def summary(self) -> str:
        lines = []
        if self.failed:
            lines.append(f"Export failed for {len(self.failed)} of {len(self.results)}:")
            lines += [f"- {r.connector}: {r.error}" for r in self.failed]
        elif not self.results:
            lines.append("Export failed: no export connector is configured")
        if self.exported:
            lines.append("Drafts created:")
            lines += [f"- {r.connector}: {r.draft_id}" for r in self.exported]
        if self.dropped:
            lines.append(f"Dropped, no draft-only endpoint: {', '.join(self.dropped)}")
        return "\n".join(lines)


def export_draft(
    text: str,
    *,
    title: str,
    approval: Approval | None,
    connectors: Iterable[Connector],
    transport: httpx.BaseTransport | None = None,
) -> ExportReport:
    """Send `text` to each connector as a draft; raises ApprovalError before any request
    unless a person approved exactly this text."""
    require_export(approval, text)
    connectors = list(connectors)
    redact = _redactor(c.api_key for c in connectors)
    report = ExportReport()
    for connector in connectors:
        if not connector.draft_only:
            log.warning("dropped export connector %s: no draft-only endpoint", connector.name)
            report.dropped.append(connector.name)
            continue
        try:
            draft_id = _create(connector, title, text, transport, redact)
        except ExportError as error:
            message = redact(str(error))
            log.error("export to %s failed: %s", connector.name, message)
            report.results.append(ExportResult(connector.name, ok=False, error=message))
        else:
            log.info("exported a draft to %s: %s", connector.name, draft_id)
            report.results.append(ExportResult(connector.name, ok=True, draft_id=draft_id))
    return report


def export_stored_draft(
    draft: Mapping[str, Any],
    *,
    connectors: Iterable[Connector],
    transport: httpx.BaseTransport | None = None,
) -> ExportReport:
    """Export a draft as the API stores it; its approval holds only for its current text."""
    stored = draft.get("approval") or {}
    sha256 = stored.get("sha256") if isinstance(stored, Mapping) else None
    approval = Approval(kind=HUMAN, sha256=sha256) if sha256 else None
    return export_draft(
        draft.get("text", ""),
        title=draft.get("title", ""),
        approval=approval,
        connectors=connectors,
        transport=transport,
    )


def _create(connector: Connector, title: str, text: str, transport, redact) -> str:
    url, headers, body = connector.request(title, text)
    try:
        with httpx.Client(timeout=connector.timeout, transport=transport) as client:
            response = client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        raise ExportError(f"timed out after {connector.timeout:g}s") from None
    except httpx.HTTPError as exc:
        raise ExportError(redact(f"request failed: {type(exc).__name__}: {exc}")) from None
    if response.status_code >= 300:
        # Some platforms echo the key back in the error body.
        body_text = redact(response.text)[:ERROR_BODY_LIMIT]
        raise ExportError(f"HTTP {response.status_code}: {body_text}")
    try:
        data = response.json()
    except ValueError:
        raise ExportError("response is not JSON") from None
    return connector.draft_id(data)


def _redactor(keys: Iterable[str]):
    """Replace every key in a message, longest first so one inside another goes whole."""
    keys = sorted({k for k in keys if k}, key=len, reverse=True)

    def redact(text: str) -> str:
        for key in keys:
            text = text.replace(key, "[redacted]")
        return text

    return redact
