"""The sample draft for a review instance: the committed `open-models-week` eval replay.

`serve --seed-sample` writes it as one draft (its text, flagged claims and run report)
into the data dir at start. Every model step is answered from the recorded responses,
so the seed needs no model key and no network. It is skipped whenever any draft already
exists: a restart never adds a second, and an existing draft is never overwritten or
re-approved.
"""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from newsletter.api.store import DraftStore, RunStore, new_id, now
from newsletter.editor import check_claims, edit_issue, summarize_stories
from newsletter.evals.replay import ReplayClient
from newsletter.evals.run import CASES_DIR, Case, load_cases
from newsletter.llm import Settings
from newsletter.render import Newsletter, build_report, render_markdown
from newsletter.sources.feed import parse_feed

SAMPLE_CASE = "open-models-week"


def seed_sample(data_dir: Path) -> bool:
    """Write the sample as one draft with its run record, unless any draft exists.
    True when the draft was written."""
    drafts = DraftStore(data_dir)
    if drafts.list():
        return False
    case = next(c for c in load_cases() if c.name == SAMPLE_CASE)
    meta = json.loads((CASES_DIR / SAMPLE_CASE / "case.json").read_text(encoding="utf-8"))
    content = _replay(case, date.fromisoformat(meta["date"]))
    stamp = now()
    run = {
        "id": new_id(),
        "status": "done",
        "error": None,
        "draft_id": None,
        "feeds": [case.feed_url],
        "title": case.title,
        "created_at": stamp,
        "started_at": stamp,
        "finished_at": stamp,
    }
    draft = drafts.create(content, run_id=run["id"])
    run["draft_id"] = draft["id"]
    RunStore(data_dir).save(run)
    return True


def _replay(case: Case, issue_date: date) -> dict[str, Any]:
    """The draft content the recorded answers produce, through the same claims, check
    and edit steps as a live run. No voice pass: the replay records none."""
    llm = ReplayClient(Settings(models=case.models), case.recorded)
    try:
        items = parse_feed(case.feed, case.feed_url).items
        stories = {f"s{n}": [item] for n, item in enumerate(items, start=1)}
        edition = edit_issue(check_claims(summarize_stories(stories, llm), llm), llm)
        report = build_report([], edition, stories, llm.report())
    finally:
        llm.close()
    newsletter = Newsletter(case.title, issue_date, edition, stories, report)
    return {
        "title": case.title,
        "issue_date": issue_date.isoformat(),
        "text": render_markdown(newsletter),
        "flags": [asdict(flag) for flag in report.flagged_claims],
        "report": report.to_dict(),
    }
