"""The visitor flow through the real API (epic 09.1 T4).

The demo page's browser test runs against a fake API; this walks the same flow through
the real one, the real pipeline, a fake model and fake feeds, so the two cannot drift:
the visitor's key headers, a check of each feed, a run with a feed that failed its check,
every progress step done, the focus in the editor prompt, flags with explanations,
approve, and the email preview.

Every call sends exactly what the page's `createApi` (`web/src/api.ts`) sends: the
visitor's `X-Model-*` headers, the owner token once it has one, and, on a write, the
`Origin` and JSON content type the browser adds. No `Authorization`: in demo mode the
visitor's credential is what authenticates the page, and with demo mode off the same
headers are refused.
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
CODE_HEADERS = {"X-Model-Preset": "demo", "X-Demo-Code": "demo-code-0123456789"}


def page(model=KEY_HEADERS, owner=None):
    """The headers `createApi` sends: the model credential and the owner token, if any.
    The browser adds `Origin` to a write; `TestClient` adds the JSON content type."""
    headers = dict(model)
    if owner:
        headers["X-Owner-Token"] = owner
    return headers


def write(model=KEY_HEADERS, owner=None):
    return {**page(model, owner), "Origin": ORIGIN}


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


def demo_app(tmp_path, *, demo=True, seen=None):
    env = {
        **ENV,
        "NEWSLETTER_API_TOKEN": TOKEN,
        "NEWSLETTER_DATA_DIR": str(tmp_path / "data"),
        "NEWSLETTER_DEMO": "1" if demo else "0",
    }
    settings = ApiSettings.from_env(env)
    transport = httpx.MockTransport(seen or Model())
    runner = default_runner(settings.env, transport=transport, fetch=web)
    return create_app(settings, runner=runner, fetch=web, resolve=public, seed_sample=True)


def test_the_visitor_flow_runs_through_the_real_api(tmp_path):
    seen = Model()
    app = demo_app(tmp_path, seen=seen)

    with TestClient(app) as client:
        # The page lists the providers before it holds a key.
        providers = {p["id"] for p in client.get("/api/providers").json()}
        assert "openrouter" in providers

        # The draft list, before the visitor has made one: the seeded example only.
        listed = client.get("/api/drafts", headers=page())
        assert listed.status_code == 200, listed.text
        assert len(listed.json()) == 1

        # Each feed is checked as the visitor adds it.
        good = client.post("/api/feeds/check", json={"url": RSS}, headers=write())
        bad = client.post("/api/feeds/check", json={"url": BROKEN}, headers=write())
        assert good.status_code == 200, good.text
        assert good.json()["ok"] is True and good.json()["items"] > 0
        assert bad.status_code == 200, bad.text
        assert bad.json()["ok"] is False and bad.json()["error"]

        # The run, on the visitor's key, with the feed that failed its check.
        started = client.post(
            "/api/runs",
            json={"feeds": [RSS, BROKEN], "title": "Visitor issue", "focus": FOCUS},
            headers=write(),
        )
        assert started.status_code == 202, started.text
        token = started.json()["owner_token"]
        owner = page(owner=token)
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
        listed = client.get("/api/drafts", headers=owner).json()
        assert draft["id"] in {d["id"] for d in listed}
        edited = client.put(
            f"/api/drafts/{draft['id']}",
            json={"text": draft["text"] + "\n"},
            headers=write(owner=token),
        )
        assert edited.status_code == 200, edited.text
        sha256 = text_sha256(edited.json()["text"])
        approved = client.post(
            f"/api/drafts/{draft['id']}/approve",
            json={"sha256": sha256},
            headers=write(owner=token),
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["approved"] is True
        email = client.get(f"/api/drafts/{draft['id']}/email", headers=owner)
        assert email.status_code == 200, email.text
        assert "Visitor issue" in email.json()["html"]

    stored = "\n".join(p.read_text("utf-8") for p in (tmp_path / "data").rglob("*") if p.is_file())
    assert KEY not in stored


def page_routes(client):
    """Every route the page calls, each with a body a write route accepts: the seeded
    example's draft and run stand in for the visitor's."""
    (example,) = client.get("/api/drafts", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    draft, run = example["id"], example["run_id"]
    return [
        ("GET", "/api/drafts", None),
        ("GET", f"/api/drafts/{draft}", None),
        ("GET", f"/api/runs/{run}", None),
        ("GET", f"/api/drafts/{draft}/email", None),
        ("PUT", f"/api/drafts/{draft}", {"text": "changed"}),
        ("POST", f"/api/drafts/{draft}/approve", {}),
        ("POST", "/api/feeds/check", {"url": RSS}),
        ("POST", "/api/runs", {"feeds": [RSS]}),
    ]


def call(client, method, path, body, headers):
    if body is None:
        return client.request(method, path, headers=headers)
    return client.request(method, path, json=body, headers={**headers, "Origin": ORIGIN})


def test_the_page_headers_pass_the_guard_on_every_route_in_demo_mode(tmp_path):
    # Not 401 on any route, with a key or a demo code. The example cannot be changed
    # (403) and is not approved (409); a run on a demo code with no gateway is 503.
    with TestClient(demo_app(tmp_path)) as client:
        for model in (KEY_HEADERS, CODE_HEADERS):
            for method, path, body in page_routes(client):
                answer = call(client, method, path, body, page(model))
                assert answer.status_code != 401, (model["X-Model-Preset"], method, path)
                assert answer.status_code in (200, 202, 403, 409, 503), (method, path, answer.text)
                if answer.status_code == 202:
                    assert client.app.state.jobs.wait(answer.json()["id"], 30)


def test_the_page_headers_are_refused_with_demo_mode_off(tmp_path):
    with TestClient(demo_app(tmp_path, demo=False)) as client:
        for model in (KEY_HEADERS, CODE_HEADERS):
            for method, path, body in page_routes(client):
                answer = call(client, method, path, body, page(model))
                assert answer.status_code == 401, (method, path, answer.text)
                assert answer.json() == {"detail": "missing or wrong token"}


def test_demo_mode_refuses_a_request_with_no_credential_or_a_malformed_one(tmp_path):
    malformed = [
        {},
        {"X-Model-Preset": "openrouter"},
        {"X-Model-Key": KEY},
        {"X-Demo-Code": "demo-code-0123456789"},
        {"X-Model-Preset": "openrouter", "X-Model-Key": "has a space"},
        {"X-Model-Preset": "nobody", "X-Model-Key": KEY},
        {"X-Model-Preset": "demo", "X-Model-Key": KEY},
        {"X-Model-Preset": "openrouter", "X-Demo-Code": "demo-code-0123456789"},
        {**KEY_HEADERS, "X-Demo-Code": "demo-code-0123456789"},
    ]
    with TestClient(demo_app(tmp_path)) as client:
        for headers in malformed:
            for method, path, body in page_routes(client):
                answer = call(client, method, path, body, headers)
                assert answer.status_code == 401, (headers, method, path, answer.text)
                assert KEY not in answer.text
        # The seeded example's run only: no request above started one.
        assert len(list((tmp_path / "data" / "runs").glob("*.json"))) == 1


def test_the_page_headers_keep_the_write_checks_in_demo_mode(tmp_path):
    with TestClient(demo_app(tmp_path)) as client:
        foreign = client.post(
            "/api/feeds/check",
            json={"url": RSS},
            headers={**page(), "Origin": "https://elsewhere.example"},
        )
        assert foreign.status_code == 403, foreign.text
        no_origin = client.post("/api/feeds/check", json={"url": RSS}, headers=page())
        assert no_origin.status_code == 403, no_origin.text
        as_text = client.post(
            "/api/feeds/check",
            content=f'{{"url": "{RSS}"}}',
            headers={**write(), "Content-Type": "text/plain"},
        )
        assert as_text.status_code == 415, as_text.text
        not_object = client.post("/api/feeds/check", json=[RSS], headers=write())
        assert not_object.status_code == 400, not_object.text


def test_the_provider_list_and_the_operator_token_still_work(tmp_path):
    operator = {"Authorization": f"Bearer {TOKEN}"}
    for demo in (True, False):
        with TestClient(demo_app(tmp_path / str(demo), demo=demo)) as client:
            assert client.get("/api/providers").status_code == 200
            assert client.get("/api/presets").status_code == 200
            listed = client.get("/api/drafts", headers=operator)
            assert listed.status_code == 200, listed.text
            assert len(listed.json()) == 1
            checked = client.post(
                "/api/feeds/check", json={"url": RSS}, headers={**operator, "Origin": ORIGIN}
            )
            assert checked.status_code == 200, checked.text
            assert checked.json()["ok"] is True
