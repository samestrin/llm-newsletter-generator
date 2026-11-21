"""Live progress of a run: the start and end of each stage, reported to a callback.

A stage is reported as `running`, then `done` or `failed`. `detail` is built in code from
counts, never from a model's reply or a source's text, so it can be shown as it is.
"""

from typing import Literal, Protocol

STEPS = [
    ("feeds", "Fetching feeds"),
    ("articles", "Reading articles"),
    ("grouping", "Grouping and ranking stories"),
    ("claims", "Writing claims"),
    ("check", "Checking claims against sources"),
    ("editor", "Editing the issue"),
    ("done", "Done"),
]
"""Every step of a run, by id and label, in the order they run."""

State = Literal["waiting", "running", "done", "failed", "skipped"]


class Progress(Protocol):
    def __call__(
        self,
        step: str,
        state: State,
        *,
        detail: str | None = None,
        done: int | None = None,
        total: int | None = None,
    ) -> None: ...


def no_progress(step: str, state: State, **counts: object) -> None:
    """The callback of a run nobody watches: the CLI and the evals."""


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    """`1 story`, `3 stories`: a count and its noun, for a `detail`."""
    return f"{count} {noun}" if count == 1 else f"{count} {nouns or noun + 's'}"


def waiting_steps() -> list[dict[str, object]]:
    """The progress of a run not started yet: every step waiting."""
    return [
        {
            "id": step,
            "label": label,
            "state": "waiting",
            "done": None,
            "total": None,
            "detail": None,
            "started_at": None,
            "ended_at": None,
        }
        for step, label in STEPS
    ]


def end_steps(steps: list[dict[str, object]], stamp: str) -> None:
    """A run has ended: a step still running failed, and a step still waiting never ran."""
    for step in steps:
        if step["state"] == "running":
            step.update(state="failed", ended_at=stamp)
        elif step["state"] == "waiting":
            step["state"] = "skipped"
