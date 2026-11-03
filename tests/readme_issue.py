"""The issue in the README's screenshots: the open-models-week eval case, replayed and
rendered. The planted false claims show up flagged.

- `web/e2e/fixtures/issue.html`: the issue as the HTML email.
- `web/e2e/fixtures/draft.json`: the issue as a draft from `newsletter serve` (the same
  fields `run_issue` returns), which the browser test seeds into the fake API.

`uv run python tests/readme_issue.py` rewrites both; `tests/test_readme.py` fails while
a committed fixture differs from what this makes now.
"""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from newsletter.editor import check_claims, edit_issue, summarize_stories
from newsletter.evals.replay import ReplayClient
from newsletter.evals.run import CASES_DIR, load_cases
from newsletter.llm import Settings
from newsletter.render import Newsletter, build_report, render_html, render_markdown
from newsletter.sources.feed import parse_feed

CASE = "open-models-week"
FIXTURES = Path(__file__).resolve().parent.parent / "web" / "e2e" / "fixtures"


def render() -> dict[str, str]:
    """Each fixture's file name and its text."""
    [case] = [c for c in load_cases(CASES_DIR) if c.name == CASE]
    meta = json.loads((CASES_DIR / CASE / "case.json").read_text(encoding="utf-8"))
    issue_date = date.fromisoformat(meta["date"])
    llm = ReplayClient(Settings(models=case.models), case.recorded)
    try:
        items = parse_feed(case.feed, case.feed_url).items
        stories = {f"s{n}": [item] for n, item in enumerate(items, start=1)}
        edition = edit_issue(check_claims(summarize_stories(stories, llm), llm), llm)
        report = build_report([], edition, stories, llm.report())
    finally:
        llm.close()
    newsletter = Newsletter(case.title, issue_date, edition, stories, report)
    draft = {
        "title": case.title,
        "issue_date": issue_date.isoformat(),
        "text": render_markdown(newsletter),
        "flags": [asdict(flag) for flag in report.flagged_claims],
        "report": report.to_dict(),
    }
    return {
        "issue.html": render_html(newsletter),
        "draft.json": json.dumps(draft, indent=2, ensure_ascii=False) + "\n",
    }


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, text in render().items():
        (FIXTURES / name).write_text(text, encoding="utf-8")
        print(FIXTURES / name)
