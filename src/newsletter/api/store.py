"""Drafts and runs, each one JSON file in the data dir, so they survive a restart.

A draft is the rendered issue text with its flags and run report. Approving it stores
the sha256 of its text; any edit clears the approval, and `is_approved` holds only while
the stored sha256 still matches the text. Files are written to a temp file and renamed
into place, so a crash leaves the old file or the new one, never half of one.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ID = re.compile(r"[0-9a-f]{32}")
REDACTED = "[redacted]"


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_approved(draft: dict[str, Any]) -> bool:
    """True only when the draft has an approval and it is of the text as it is now."""
    approval = draft.get("approval")
    return bool(approval) and approval.get("sha256") == text_sha256(draft.get("text", ""))


def redactor(secrets: Iterable[str]) -> Callable[[Any], Any]:
    """Replace every secret in a value's strings, however deeply nested."""
    secrets = [s for s in secrets if s]

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, REDACTED)
            return value
        if isinstance(value, dict):
            return {redact(k): redact(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [redact(v) for v in value]
        return value

    return redact


class NotFound(KeyError):
    pass


class Conflict(ValueError):
    pass


class JsonFiles:
    """One JSON object per id in a folder."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()

    def path(self, item_id: str) -> Path:
        if not _ID.fullmatch(item_id):
            raise NotFound(item_id)
        return self.folder / f"{item_id}.json"

    def read(self, item_id: str) -> dict[str, Any]:
        path = self.path(item_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise NotFound(item_id) from None

    def write(self, item_id: str, data: dict[str, Any]) -> None:
        path = self.path(item_id)
        self.folder.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.folder, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False, indent=2)
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    def all(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.folder.glob("*.json")):
            if not _ID.fullmatch(path.stem):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                log.warning("skipped unreadable file %s: %s", path, type(error).__name__)
                continue
            if isinstance(data, dict):
                items.append(data)
            else:
                log.warning("skipped unreadable file %s: not an object", path)
        return items


class DraftStore:
    def __init__(self, data_dir: Path):
        self.files = JsonFiles(data_dir / "drafts")

    def create(self, content: dict[str, Any], *, run_id: str) -> dict[str, Any]:
        stamp = now()
        draft = {
            "id": new_id(),
            "run_id": run_id,
            "title": content["title"],
            "issue_date": content["issue_date"],
            "text": content["text"],
            "flags": content.get("flags", []),
            "report": content.get("report", {}),
            "approval": None,
            "created_at": stamp,
            "updated_at": stamp,
        }
        with self.files.lock:
            self.files.write(draft["id"], draft)
        return draft

    def get(self, draft_id: str) -> dict[str, Any]:
        return self.files.read(draft_id)

    def list(self) -> list[dict[str, Any]]:
        drafts = self.files.all()
        return sorted(drafts, key=lambda d: d.get("created_at", ""), reverse=True)

    def edit(self, draft_id: str, text: str) -> dict[str, Any]:
        """Replace the text. Any edit clears the approval, even one to the same text."""
        with self.files.lock:
            draft = self.files.read(draft_id)
            draft.update(text=text, approval=None, updated_at=now())
            self.files.write(draft_id, draft)
        return draft

    def approve(self, draft_id: str, expected_sha256: str | None = None) -> dict[str, Any]:
        """Approve the text as it is now; with `expected_sha256`, only if it is that text."""
        with self.files.lock:
            draft = self.files.read(draft_id)
            digest = text_sha256(draft["text"])
            if expected_sha256 is not None and expected_sha256 != digest:
                raise Conflict("the draft changed since it was read; reload it and approve again")
            stamp = now()
            draft.update(approval={"sha256": digest, "approved_at": stamp}, updated_at=stamp)
            self.files.write(draft_id, draft)
        return draft


class RunStore:
    def __init__(self, data_dir: Path):
        self.files = JsonFiles(data_dir / "runs")

    def get(self, run_id: str) -> dict[str, Any]:
        return self.files.read(run_id)

    def save(self, run: dict[str, Any]) -> None:
        with self.files.lock:
            self.files.write(run["id"], run)

    def mark_interrupted(self) -> None:
        """A run still open from an earlier server was cut off by its restart."""
        for run in self.files.all():
            if run.get("status") in {"queued", "running"} and _ID.fullmatch(run.get("id", "")):
                run.update(
                    status="error",
                    error="the server restarted before the run finished",
                    finished_at=now(),
                )
                self.save(run)
