"""Runs as background jobs, each with a status and a timeout.

A run goes `queued`, `running`, then `done` with its draft id or `error` with why. A
thread cannot be killed, so a run past its timeout is reported as an error at once and
whatever it returns later is thrown away; no draft appears for a run reported as failed.
Each model and feed request has its own timeout, so the thread itself ends soon after.
One run at a time, counting a timed-out one still finishing, so runs cannot pile up.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from newsletter.api.pipeline import RunRequest
from newsletter.api.store import DraftStore, RunStore, new_id, now

log = logging.getLogger(__name__)

Runner = Callable[[RunRequest], dict[str, Any]]


class Busy(RuntimeError):
    pass


class Jobs:
    def __init__(
        self,
        runner: Runner,
        runs: RunStore,
        drafts: DraftStore,
        *,
        timeout: float,
        redact: Callable[[Any], Any],
    ):
        self.runner = runner
        self.runs = runs
        self.drafts = drafts
        self.timeout = timeout
        self.redact = redact
        self._lock = threading.Lock()
        self._finished: dict[str, threading.Event] = {}
        self._idle = threading.Event()
        self._idle.set()

    def start(self, request: RunRequest) -> dict[str, Any]:
        with self._lock:
            if not self._idle.is_set():
                raise Busy("a run is already going; wait for it to finish")
            self._idle.clear()
            run = {
                "id": new_id(),
                "status": "queued",
                "error": None,
                "draft_id": None,
                "feeds": list(request.feeds),
                "title": request.title,
                "created_at": now(),
                "started_at": None,
                "finished_at": None,
            }
            self.runs.save(run)
            self._finished[run["id"]] = threading.Event()
        threading.Thread(target=self._watch, args=(run, request), daemon=True).start()
        return run

    def wait(self, run_id: str, timeout: float | None = None) -> bool:
        """Wait for a run to end, as done or as an error. False if it is still going."""
        event = self._finished.get(run_id)
        return True if event is None else event.wait(timeout)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until no run thread is left, a timed-out one included."""
        return self._idle.wait(timeout)

    def _watch(self, run: dict[str, Any], request: RunRequest) -> None:
        outcome: dict[str, Any] = {}
        lock = threading.Lock()

        def work() -> None:
            try:
                content = self.redact(self.runner(request))
                with lock:
                    if outcome:  # Timed out already: its result is not used.
                        return
                    draft = self.drafts.create(content, run_id=run["id"])
                    outcome.update(status="done", draft_id=draft["id"])
            except Exception as error:
                message = self.redact(f"{type(error).__name__}: {error}")
                with lock:
                    if outcome:
                        log.warning("run %s failed after its timeout: %s", run["id"], message)
                        return
                    outcome.update(status="error", error=message)
                log.error("run %s failed: %s", run["id"], message)
            finally:
                worker_done.set()
                self._idle.set()

        worker_done = threading.Event()
        run.update(status="running", started_at=now())
        self.runs.save(run)
        threading.Thread(target=work, daemon=True).start()
        if not worker_done.wait(self.timeout):
            with lock:
                if not outcome:
                    message = (
                        f"the run took longer than {self.timeout:g} seconds; its result is not used"
                    )
                    outcome.update(status="error", error=message)
                    log.error("run %s: %s", run["id"], message)
        with lock:
            run.update(outcome, finished_at=now())
            self.runs.save(run)
        self._finished[run["id"]].set()
