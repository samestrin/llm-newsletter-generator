"""The review page's contract fixture is the run report the API really serves.

`web/src/contract/seed-report.json` is the `report` of the `--seed-sample` draft, as
`GET /api/drafts/{id}` returns it. The web tests render that file, so a field the page
reads and the API never sends shows up there. When the report changes on purpose,
rewrite the file from this test's `served_report()` output.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from newsletter.api import ApiSettings, create_app

FIXTURE = Path(__file__).parent.parent / "web" / "src" / "contract" / "seed-report.json"
TOKEN = "contract-token"


def served_report(tmp_path) -> dict:
    env = {"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_DATA_DIR": str(tmp_path / "data")}
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(create_app(ApiSettings.from_env(env), seed_sample=True)) as client:
        (summary,) = client.get("/api/drafts", headers=headers).json()
        return client.get(f"/api/drafts/{summary['id']}", headers=headers).json()["report"]


def test_the_web_fixture_is_the_report_the_api_serves(tmp_path):
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == served_report(tmp_path)


def test_the_served_steps_carry_token_counts(tmp_path):
    steps = served_report(tmp_path)["steps"]
    assert steps
    for step in steps:
        assert isinstance(step["prompt_tokens"], int)
        assert isinstance(step["completion_tokens"], int)
