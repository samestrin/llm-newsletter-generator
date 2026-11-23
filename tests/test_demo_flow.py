"""The visitor flow through the real API (epic 09.1 T4).

The demo page's browser test runs against a fake API; this walks the same flow through
the real one, the real pipeline, a fake model and fake feeds, so the two cannot drift:
the visitor's key headers, a check of each feed, a run with a feed that failed its check,
every progress step done, the focus in the editor prompt, flags with explanations,
approve, and the email preview.
"""

import json

import httpx
from fastapi.testclient import TestClient

from newsletter.api import ApiSettings, create_app
from newsletter.api.pipeline import default_runner
from newsletter.api.store import text_sha256
from test_api import ORIGIN, TOKEN
from test_cli import ENV, FALSE_CLAIM, RSS, fixture_web, model
from test_feed_check import BROKEN, Web, public

KEY = "sk-visitor-flow-key-0123456789"
FOCUS = "open models anyone can run"
KEY_HEADERS = {"X-Model-Preset": "openrouter", "X-Model-Key": KEY}


def web(url, headers, timeout):
    """The fixture feed, and the feed-check tests' broken feed."""
    if url.startswith("https://broken.example.org/"):
        return Web()(url, headers, timeout)
    return fixture_web(url, headers, timeout)


class Model:
    """The fake model of `test_cli`, keeping the bearer of each call and each editor
    prompt."""

    def __init__(self):
        self.bearers = []
        self.editor_prompts = []
        self.answer = model().handler

    def __call__(self, request):
        self.bearers.append(request.headers.get("authorization"))
        body = json.loads(request.content)
        if (body.get("response_format") or {}).get("json_schema", {}).get("name") == "issue_edit":
            self.editor_prompts.append(body["messages"][-1]["content"])
        return self.answer(request)


def test_the_visitor_flow_runs_through_the_real_api(tmp_path):
    env = {
        **ENV,
        "NEWSLETTER_API_TOKEN": TOKEN,
        "NEWSLETTER_DATA_DIR": str(tmp_path / "data"),
        "NEWSLETTER_DEMO": "1",
    }
    settings = ApiSettings.from_env(env)
    seen = Model()
    runner = default_runner(settings.env, transport=httpx.MockTransport(seen), fetch=web)
    app = create_app(settings, runner=runner, fetch=web, resolve=public)
    sent = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN}

    with TestClient(app) as client:
        # The page lists the providers before it holds a key.
        providers = {p["id"] for p in client.get("/api/providers").json()}
        assert "openrouter" in providers

        # Each feed is checked as the visitor adds it.
        good = client.post("/api/feeds/check", json={"url": RSS}, headers=sent)
        bad = client.post("/api/feeds/check", json={"url": BROKEN}, headers=sent)
        assert good.status_code == 200, good.text
        assert good.json()["ok"] is True and good.json()["items"] > 0
        assert bad.status_code == 200, bad.text
        assert bad.json()["ok"] is False and bad.json()["error"]

        # The run, on the visitor's key, with the feed that failed its check.
        started = client.post(
            "/api/runs",
            json={"feeds": [RSS, BROKEN], "title": "Visitor issue", "focus": FOCUS},
            headers={**sent, **KEY_HEADERS},
        )
        assert started.status_code == 202, started.text
        owner = {**sent, "X-Owner-Token": started.json()["owner_token"]}
        run_id = started.json()["id"]
        assert client.app.state.jobs.wait(run_id, 30)

        run = client.get(f"/api/runs/{run_id}", headers=owner).json()
        assert run["status"] == "done", run
        assert [step["state"] for step in run["progress"]] == ["done"] * len(run["progress"])
        feeds = next(step for step in run["progress"] if step["label"] == "Fetching feeds")
        assert f"1 failed: {BROKEN} {bad.json()['error']}" in feeds["detail"]

        # Every model call went with the visitor's key, and the editor was told the focus.
        assert seen.bearers and set(seen.bearers) == {f"Bearer {KEY}"}
        (prompt,) = seen.editor_prompts
        assert f"<focus>\n{FOCUS}\n</focus>" in prompt

        draft = client.get(f"/api/drafts/{run['draft_id']}", headers=owner).json()
        assert draft["focus"] == FOCUS
        failed = {(f["url"], f["reason"]) for f in draft["report"]["failed_sources"]}
        assert (BROKEN, bad.json()["error"]) in failed
        flagged = [flag for flag in draft["flags"] if flag["text"] == FALSE_CLAIM]
        assert flagged, draft["flags"]
        assert all(flag.get("explanation") for flag in draft["flags"])

        # No preview before approval; approve the text as read, then preview it.
        early = client.get(f"/api/drafts/{draft['id']}/email", headers=owner)
        assert early.status_code == 409, early.text
        sha256 = text_sha256(draft["text"])
        approved = client.post(
            f"/api/drafts/{draft['id']}/approve", json={"sha256": sha256}, headers=owner
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["approved"] is True
        email = client.get(f"/api/drafts/{draft['id']}/email", headers=owner)
        assert email.status_code == 200, email.text
        assert "Visitor issue" in email.json()["html"]

    stored = "\n".join(p.read_text("utf-8") for p in (tmp_path / "data").rglob("*") if p.is_file())
    assert KEY not in stored
