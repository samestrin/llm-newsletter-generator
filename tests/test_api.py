import hashlib
import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from newsletter.api import ApiSettings, create_app
from newsletter.api.store import is_approved
from newsletter.config import HackerNewsConfig, SourcesConfig

TOKEN = "api-token-not-for-anyone-else"
LLM_KEY = "sk-llm-secret-456"
VOICE_KEY = "yv-secret-789"
ORIGIN = "http://127.0.0.1:8000"
TEXT = "# Weekly\n\nA story [1].\n"
CONTENT = {
    "title": "Weekly",
    "issue_date": "2026-11-02",
    "text": TEXT,
    "flags": [{"id": "s1.c2", "story_id": "s1", "text": "A claim", "status": "unsupported"}],
    "report": {"ok": False, "flagged_claims": [{"id": "s1.c2"}]},
}
RUN = {"feeds": ["https://example.org/feed.xml"], "title": "Weekly", "topic": "AI"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def headers(**extra) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN, **extra}


def make_settings(tmp_path, **changes) -> ApiSettings:
    env = {
        "NEWSLETTER_API_TOKEN": TOKEN,
        "NEWSLETTER_DATA_DIR": str(tmp_path / "data"),
        "NEWSLETTER_API_KEY": LLM_KEY,
        "YOURVOICE_API_KEY": VOICE_KEY,
        **changes,
    }
    return ApiSettings.from_env(env)


class FakeRunner:
    """Stands in for the editor, voice and render pipeline."""

    def __init__(self, result=CONTENT, *, block: threading.Event | None = None):
        self.result = result
        self.block = block
        self.requests = []

    def __call__(self, request, progress=None):
        self.requests.append(request)
        if self.block is not None:
            self.block.wait(10)
        if isinstance(self.result, Exception):
            raise self.result
        return dict(self.result)


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def client(tmp_path, runner):
    app = create_app(make_settings(tmp_path), runner=runner)
    with TestClient(app) as client:
        yield client


def run_and_wait(client, body=RUN) -> dict:
    response = client.post("/api/runs", json=body, headers=headers())
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["status"] in {"queued", "running", "done", "error"}
    assert client.app.state.jobs.wait(run["id"], 5)
    return client.get(f"/api/runs/{run['id']}", headers=headers()).json()


def make_draft(client) -> dict:
    run = run_and_wait(client)
    assert run["status"] == "done", run
    return client.get(f"/api/drafts/{run['draft_id']}", headers=headers()).json()


# Every write is refused without the token, a matching Origin and a JSON body.

WRITES = [
    ("post", "/api/runs", RUN),
    ("put", "/api/drafts/{id}", {"text": "changed"}),
    ("post", "/api/drafts/{id}/approve", {}),
]


@pytest.mark.parametrize("method,path,body", WRITES)
@pytest.mark.parametrize(
    "auth",
    [None, "Bearer wrong-token", f"Basic {TOKEN}", f"Bearer {TOKEN}x", "Bearer "],
)
def test_a_write_without_the_token_is_refused(client, runner, method, path, body, auth):
    draft = make_draft(client)
    sent = {"Origin": ORIGIN} if auth is None else {"Origin": ORIGIN, "Authorization": auth}
    response = client.request(method, path.format(id=draft["id"]), json=body, headers=sent)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in response.text
    assert len(runner.requests) == 1
    assert client.get(f"/api/drafts/{draft['id']}", headers=headers()).json() == draft


@pytest.mark.parametrize("method,path,body", WRITES)
@pytest.mark.parametrize(
    "origin", [None, "https://evil.example", "null", "http://127.0.0.1:8001", ORIGIN + "/"]
)
def test_a_write_with_a_wrong_origin_is_refused(client, runner, method, path, body, origin):
    draft = make_draft(client)
    sent = {"Authorization": f"Bearer {TOKEN}"}
    if origin is not None:
        sent["Origin"] = origin
    response = client.request(method, path.format(id=draft["id"]), json=body, headers=sent)
    assert response.status_code == 403
    assert len(runner.requests) == 1
    assert client.get(f"/api/drafts/{draft['id']}", headers=headers()).json() == draft


@pytest.mark.parametrize("method,path,body", WRITES)
@pytest.mark.parametrize(
    "content_type,raw,status",
    [
        ("text/plain", "{}", 415),
        ("application/x-www-form-urlencoded", "text=changed", 415),
        ("multipart/form-data; boundary=x", "--x--", 415),
        (None, "{}", 415),
        ("application/json", "not json", 400),
        ("application/json", "[1, 2]", 400),
        ("application/json", "", 400),
    ],
)
def test_a_write_with_a_non_json_body_is_refused(
    client, runner, method, path, body, content_type, raw, status
):
    draft = make_draft(client)
    sent = headers()
    if content_type:
        sent["Content-Type"] = content_type
    response = client.request(method, path.format(id=draft["id"]), content=raw, headers=sent)
    assert response.status_code == status
    assert len(runner.requests) == 1
    assert client.get(f"/api/drafts/{draft['id']}", headers=headers()).json() == draft


def test_a_json_content_type_with_a_charset_is_accepted(client):
    draft = make_draft(client)
    response = client.put(
        f"/api/drafts/{draft['id']}",
        content=json.dumps({"text": "changed"}),
        headers=headers(**{"Content-Type": "application/json; charset=utf-8"}),
    )
    assert response.status_code == 200
    assert response.json()["text"] == "changed"


def test_writes_are_refused_when_no_token_is_set(tmp_path, runner):
    settings = make_settings(tmp_path, NEWSLETTER_API_TOKEN="")
    with TestClient(create_app(settings, runner=runner)) as client:
        response = client.post("/api/runs", json=RUN, headers=headers())
    assert response.status_code == 503
    assert "NEWSLETTER_API_TOKEN" in response.json()["detail"]
    assert runner.requests == []


def test_reads_need_the_token_too(client):
    draft = make_draft(client)
    for path in ("/api/drafts", f"/api/drafts/{draft['id']}"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/health").json() == {"ok": True}


def test_a_body_that_fails_validation_is_refused_without_echoing_it(client):
    draft = make_draft(client)
    response = client.put(
        f"/api/drafts/{draft['id']}", json={"text": 5, "note": TOKEN}, headers=headers()
    )
    assert response.status_code == 422
    assert TOKEN not in response.text
    assert "text" in response.text


def test_a_run_needs_at_least_one_feed(client, runner):
    for body in ({"feeds": []}, {"title": "Weekly"}):
        response = client.post("/api/runs", json=body, headers=headers())
        assert response.status_code == 422
    assert runner.requests == []


def test_a_run_with_no_topic_and_no_config_is_about_news(client, runner):
    run_and_wait(client, {"feeds": ["https://example.org/feed.xml"]})
    assert runner.requests[0].topic == "news"


CONFIG = SourcesConfig(
    "open models",
    feeds=("https://example.org/feed.xml",),
    pages=("https://example.org/article",),
    hackernews=HackerNewsConfig(min_score=50),
    per_feed=2,
)


def test_a_run_with_no_feeds_reads_the_sources_config(tmp_path, runner):
    app = create_app(make_settings(tmp_path), runner=runner, config=CONFIG)
    with TestClient(app) as client:
        run = run_and_wait(client, {"title": "Weekly"})
    assert run["status"] == "done"
    (request,) = runner.requests
    assert request.feeds == ["https://example.org/feed.xml"]
    assert request.pages == ["https://example.org/article"]
    assert request.hackernews == HackerNewsConfig(min_score=50)
    assert request.per_feed == 2
    assert request.topic == "open models"


def test_feeds_in_the_run_win_over_the_sources_config(tmp_path, runner):
    app = create_app(make_settings(tmp_path), runner=runner, config=CONFIG)
    with TestClient(app) as client:
        run_and_wait(client, {"feeds": ["https://b.example/feed.xml"], "topic": "AI"})
    (request,) = runner.requests
    assert request.feeds == ["https://b.example/feed.xml"]
    assert request.pages == [] and request.hackernews is None
    assert request.per_feed == 2
    assert request.topic == "AI"


def test_the_browser_page_may_preflight_from_an_allowed_origin(client):
    response = client.options(
        "/api/runs",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    refused = client.options(
        "/api/runs",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in refused.headers


# Runs, drafts, edits and approval.


def test_a_run_writes_a_draft_with_its_flags_and_report(client, runner):
    run = run_and_wait(client)
    assert run["status"] == "done"
    assert run["error"] is None
    assert runner.requests[0].feeds == RUN["feeds"]
    assert runner.requests[0].title == "Weekly"
    assert runner.requests[0].topic == "AI"

    listed = client.get("/api/drafts", headers=headers()).json()
    assert [d["id"] for d in listed] == [run["draft_id"]]
    assert listed[0]["title"] == "Weekly"
    assert listed[0]["approved"] is False
    assert "text" not in listed[0]

    draft = client.get(f"/api/drafts/{run['draft_id']}", headers=headers()).json()
    assert draft["text"] == TEXT
    assert draft["flags"] == CONTENT["flags"]
    assert draft["report"] == CONTENT["report"]
    assert draft["approval"] is None
    assert draft["run_id"] == run["id"]


def test_approval_stores_the_sha256_of_the_draft_text(client):
    draft = make_draft(client)
    response = client.post(f"/api/drafts/{draft['id']}/approve", json={}, headers=headers())
    assert response.status_code == 200
    approval = response.json()["approval"]
    assert approval["sha256"] == sha(TEXT)
    assert approval["approved_at"]
    assert response.json()["approved"] is True


def test_an_edit_after_approval_clears_the_approval(client):
    draft = make_draft(client)
    client.post(f"/api/drafts/{draft['id']}/approve", json={}, headers=headers())

    response = client.put(
        f"/api/drafts/{draft['id']}", json={"text": "# Weekly\n\nEdited.\n"}, headers=headers()
    )
    assert response.status_code == 200
    edited = response.json()
    assert edited["text"] == "# Weekly\n\nEdited.\n"
    assert edited["approval"] is None
    assert edited["approved"] is False
    stored = client.get(f"/api/drafts/{draft['id']}", headers=headers()).json()
    assert stored["approval"] is None
    listed = client.get("/api/drafts", headers=headers()).json()
    assert listed[0]["approved"] is False


def test_an_edit_that_keeps_the_text_still_clears_the_approval(client):
    draft = make_draft(client)
    client.post(f"/api/drafts/{draft['id']}/approve", json={}, headers=headers())
    edited = client.put(f"/api/drafts/{draft['id']}", json={"text": TEXT}, headers=headers())
    assert edited.json()["approval"] is None


def test_approval_of_a_text_the_reviewer_did_not_see_is_refused(client):
    draft = make_draft(client)
    response = client.post(
        f"/api/drafts/{draft['id']}/approve", json={"sha256": sha("older text")}, headers=headers()
    )
    assert response.status_code == 409
    assert client.get(f"/api/drafts/{draft['id']}", headers=headers()).json()["approval"] is None
    response = client.post(
        f"/api/drafts/{draft['id']}/approve", json={"sha256": sha(TEXT)}, headers=headers()
    )
    assert response.status_code == 200


def test_is_approved_needs_an_approval_that_matches_the_text():
    draft = {"text": TEXT, "approval": {"sha256": sha(TEXT), "approved_at": "x"}}
    assert is_approved(draft)
    assert not is_approved({**draft, "text": TEXT + "more"})
    assert not is_approved({**draft, "approval": None})
    assert not is_approved({"text": TEXT})


@pytest.mark.parametrize("draft_id", ["nope", "../../etc/passwd", "0" * 31, "A" * 32])
def test_an_unknown_draft_is_not_found(client, draft_id):
    assert client.get(f"/api/drafts/{draft_id}", headers=headers()).status_code == 404
    assert client.get(f"/api/drafts/{draft_id}/email", headers=headers()).status_code == 404
    assert (
        client.put(f"/api/drafts/{draft_id}", json={"text": "x"}, headers=headers()).status_code
        == 404
    )
    assert client.get(f"/api/runs/{draft_id}", headers=headers()).status_code == 404


def test_a_draft_and_its_approval_survive_a_restart(tmp_path, runner):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings, runner=runner)) as client:
        draft = make_draft(client)
        client.post(f"/api/drafts/{draft['id']}/approve", json={}, headers=headers())
        before = client.get(f"/api/drafts/{draft['id']}", headers=headers()).json()

    with TestClient(create_app(settings, runner=FakeRunner())) as client:
        listed = client.get("/api/drafts", headers=headers()).json()
        after = client.get(f"/api/drafts/{draft['id']}", headers=headers()).json()
        run = client.get(f"/api/runs/{draft['run_id']}", headers=headers()).json()
    assert [d["id"] for d in listed] == [draft["id"]]
    assert after == before
    assert after["approved"] is True
    assert run["status"] == "done"
    assert list((tmp_path / "data" / "drafts").glob("*.json")) == [
        tmp_path / "data" / "drafts" / f"{draft['id']}.json"
    ]


def test_a_run_cut_off_by_a_restart_is_reported_as_an_error(tmp_path):
    settings = make_settings(tmp_path)
    release = threading.Event()
    app = create_app(settings, runner=FakeRunner(block=release))
    with TestClient(app) as client:
        run = client.post("/api/runs", json=RUN, headers=headers()).json()
    # The first server is gone with its run still going.
    with TestClient(create_app(settings, runner=FakeRunner())) as client:
        status = client.get(f"/api/runs/{run['id']}", headers=headers()).json()
    release.set()
    assert status["status"] == "error"
    assert "restart" in status["error"]


def test_drafts_are_listed_newest_first(client):
    first = make_draft(client)
    second = make_draft(client)
    listed = client.get("/api/drafts", headers=headers()).json()
    assert [d["id"] for d in listed] == [second["id"], first["id"]]


def test_an_unreadable_draft_file_is_skipped_and_logged(client, tmp_path, caplog):
    draft = make_draft(client)
    (tmp_path / "data" / "drafts" / ("f" * 32 + ".json")).write_text("{broken", "utf-8")
    with caplog.at_level(logging.WARNING):
        listed = client.get("/api/drafts", headers=headers()).json()
    assert [d["id"] for d in listed] == [draft["id"]]
    assert "f" * 32 in caplog.text


# Timeouts and failures.


def test_a_run_that_exceeds_its_timeout_ends_as_a_reported_error(tmp_path):
    release = threading.Event()
    runner = FakeRunner(block=release)
    settings = make_settings(tmp_path, NEWSLETTER_RUN_TIMEOUT="0.2")
    with TestClient(create_app(settings, runner=runner)) as client:
        run = run_and_wait(client)
        assert run["status"] == "error"
        assert run["error"] == "the run took longer than 0.2 seconds; its result is not used"
        assert run["draft_id"] is None

        # The late result is thrown away: no draft appears for a run reported as failed.
        release.set()
        assert client.app.state.jobs.wait_idle(5)
        assert client.get("/api/drafts", headers=headers()).json() == []
        assert client.get(f"/api/runs/{run['id']}", headers=headers()).json() == run


def test_only_one_run_at_a_time(tmp_path):
    release = threading.Event()
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings, runner=FakeRunner(block=release))) as client:
        first = client.post("/api/runs", json=RUN, headers=headers())
        second = client.post("/api/runs", json=RUN, headers=headers())
        release.set()
        assert client.app.state.jobs.wait(first.json()["id"], 5)
    assert first.status_code == 202
    assert second.status_code == 409


def test_a_failed_run_is_reported_without_any_secret(tmp_path, caplog):
    error = RuntimeError(f"boom with {TOKEN} and {LLM_KEY} and {VOICE_KEY}")
    settings = make_settings(tmp_path)
    with caplog.at_level(logging.DEBUG):
        with TestClient(create_app(settings, runner=FakeRunner(error))) as client:
            run = run_and_wait(client)
            listed = client.get("/api/drafts", headers=headers())
            stored = (tmp_path / "data" / "runs" / f"{run['id']}.json").read_text("utf-8")
    assert run["status"] == "error"
    assert run["error"].startswith("RuntimeError: boom with ")
    assert listed.json() == []
    for secret in (TOKEN, LLM_KEY, VOICE_KEY):
        assert secret not in json.dumps(run)
        assert secret not in stored
        assert secret not in caplog.text
    assert "[redacted]" in run["error"]


def test_no_secret_appears_in_a_draft_or_its_report(tmp_path, caplog):
    leaky = {**CONTENT, "text": f"text {LLM_KEY}", "report": {"step": f"failed: {VOICE_KEY}"}}
    with caplog.at_level(logging.DEBUG):
        with TestClient(create_app(make_settings(tmp_path), runner=FakeRunner(leaky))) as client:
            run = run_and_wait(client)
            draft = client.get(f"/api/drafts/{run['draft_id']}", headers=headers())
    for secret in (TOKEN, LLM_KEY, VOICE_KEY):
        assert secret not in draft.text
        assert secret not in caplog.text


def test_the_settings_repr_holds_no_secret(tmp_path):
    settings = make_settings(tmp_path)
    assert TOKEN not in repr(settings)
    assert LLM_KEY not in repr(settings)


# Settings and the bind address.


def test_the_default_bind_is_127_0_0_1(tmp_path):
    settings = ApiSettings.from_env({})
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.origins == ("http://127.0.0.1:8000", "http://localhost:8000")
    assert settings.token is None
    assert settings.run_timeout == 900.0


def test_settings_from_the_environment(tmp_path):
    settings = ApiSettings.from_env(
        {
            "NEWSLETTER_API_TOKEN": TOKEN,
            "NEWSLETTER_API_HOST": "0.0.0.0",
            "NEWSLETTER_API_PORT": "9000",
            "NEWSLETTER_API_ORIGINS": "http://localhost:5173/, http://127.0.0.1:5173",
            "NEWSLETTER_DATA_DIR": str(tmp_path),
            "NEWSLETTER_RUN_TIMEOUT": "30",
        }
    )
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.origins == ("http://localhost:5173", "http://127.0.0.1:5173")
    assert settings.data_dir == tmp_path
    assert settings.run_timeout == 30.0
    assert settings.token == TOKEN


@pytest.mark.parametrize(
    "name,value",
    [
        ("NEWSLETTER_API_PORT", "http"),
        ("NEWSLETTER_API_PORT", "0"),
        ("NEWSLETTER_RUN_TIMEOUT", "-1"),
        ("NEWSLETTER_RUN_TIMEOUT", "0"),
    ],
)
def test_bad_settings_are_refused(name, value):
    from newsletter.llm import ConfigError

    with pytest.raises(ConfigError, match=name):
        ApiSettings.from_env({name: value})


# `newsletter serve`.


def test_serve_binds_127_0_0_1_by_default(tmp_path, monkeypatch):
    import uvicorn

    from newsletter.cli import main

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    env = {"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_DATA_DIR": str(tmp_path)}
    assert main(["serve"], env=env) == 0
    ((app, kwargs),) = calls
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert app.state.settings.host == "127.0.0.1"


def test_serve_takes_a_host_and_port(tmp_path, monkeypatch):
    import uvicorn

    from newsletter.cli import main

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    env = {"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_DATA_DIR": str(tmp_path)}
    assert main(["serve", "--host", "0.0.0.0", "--port", "9001"], env=env) == 0
    assert calls[0]["host"] == "0.0.0.0" and calls[0]["port"] == 9001


def test_serve_reads_a_sources_config(tmp_path, monkeypatch):
    import uvicorn

    from newsletter.cli import main

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(app))
    config = tmp_path / "sources.yaml"
    config.write_text("topic: open models\nsources:\n  - page: https://a.test/p\n", "utf-8")
    env = {"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_DATA_DIR": str(tmp_path)}
    assert main(["serve", "--config", str(config)], env=env) == 0
    assert calls[0].state.config == SourcesConfig("open models", pages=("https://a.test/p",))


def test_serve_refuses_an_invalid_sources_config(tmp_path, monkeypatch, capsys):
    import uvicorn

    from newsletter.cli import main

    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: pytest.fail("started"))
    config = tmp_path / "sources.yaml"
    config.write_text("topic: t\nsources: [{page: 3}]\n", "utf-8")
    env = {"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_DATA_DIR": str(tmp_path)}
    assert main(["serve", "--config", str(config)], env=env) == 2
    assert "sources[0].page" in capsys.readouterr().err


@pytest.mark.parametrize(
    "env,message",
    [
        ({}, "NEWSLETTER_API_TOKEN"),
        ({"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_API_PORT": "x"}, "NEWSLETTER_API_PORT"),
    ],
)
def test_serve_refuses_to_start_without_good_settings(monkeypatch, capsys, env, message):
    import uvicorn

    from newsletter.cli import main

    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: pytest.fail("started"))
    assert main(["serve"], env=env) == 2
    assert message in capsys.readouterr().err


# `serve --seed-sample`: the committed eval replay as one draft, no model key.


def seed_env(tmp_path) -> dict[str, str]:
    """Only the token and the data dir: no model key, no voice key."""
    return {"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_DATA_DIR": str(tmp_path / "data")}


def test_seed_sample_lists_one_draft_with_a_flagged_claim_and_a_report(tmp_path):
    settings = ApiSettings.from_env(seed_env(tmp_path))
    with TestClient(create_app(settings, seed_sample=True)) as client:
        listed = client.get("/api/drafts", headers=headers()).json()
        assert len(listed) == 1
        assert listed[0]["flags"] >= 1
        assert listed[0]["approved"] is False
        draft = client.get(f"/api/drafts/{listed[0]['id']}", headers=headers()).json()
        assert draft["text"].startswith("# Open models week")
        assert draft["flags"]
        assert draft["report"]["flagged_claims"]
        assert draft["report"]["steps"]
        run = client.get(f"/api/runs/{draft['run_id']}", headers=headers()).json()
    assert run["status"] == "done"
    assert run["draft_id"] == draft["id"]


def test_seed_sample_a_second_start_lists_still_one(tmp_path):
    settings = ApiSettings.from_env(seed_env(tmp_path))
    with TestClient(create_app(settings, seed_sample=True)) as client:
        before = client.get("/api/drafts", headers=headers()).json()
        draft_before = client.get(f"/api/drafts/{before[0]['id']}", headers=headers()).json()
    with TestClient(create_app(settings, seed_sample=True)) as client:
        after = client.get("/api/drafts", headers=headers()).json()
        draft_after = client.get(f"/api/drafts/{after[0]['id']}", headers=headers()).json()
    assert len(after) == 1
    assert draft_after == draft_before
    assert len(list((tmp_path / "data" / "drafts").glob("*.json"))) == 1
    assert len(list((tmp_path / "data" / "runs").glob("*.json"))) == 1


@pytest.mark.parametrize("approved", [False, True])
def test_seed_sample_never_overwrites_or_re_approves_an_existing_draft(tmp_path, runner, approved):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings, runner=runner)) as client:
        draft = make_draft(client)
        if approved:
            client.post(f"/api/drafts/{draft['id']}/approve", json={}, headers=headers())
        before = client.get(f"/api/drafts/{draft['id']}", headers=headers()).json()
    with TestClient(create_app(settings, runner=FakeRunner(), seed_sample=True)) as client:
        listed = client.get("/api/drafts", headers=headers()).json()
        after = client.get(f"/api/drafts/{draft['id']}", headers=headers()).json()
    assert [d["id"] for d in listed] == [draft["id"]]
    assert after == before
    assert after["approved"] is approved


def test_without_seed_sample_the_list_is_empty(client):
    assert client.get("/api/drafts", headers=headers()).json() == []


def test_serve_seed_sample_writes_the_draft_at_start(tmp_path, monkeypatch):
    import uvicorn

    from newsletter.cli import main

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(app))
    assert main(["serve", "--seed-sample"], env=seed_env(tmp_path)) == 0
    with TestClient(calls[0]) as client:
        listed = client.get("/api/drafts", headers=headers()).json()
    assert len(listed) == 1
    assert listed[0]["flags"] >= 1


def test_serve_without_seed_sample_starts_with_no_drafts(tmp_path, monkeypatch):
    import uvicorn

    from newsletter.cli import main

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(app))
    assert main(["serve"], env=seed_env(tmp_path)) == 0
    with TestClient(calls[0]) as client:
        assert client.get("/api/drafts", headers=headers()).json() == []


@pytest.mark.parametrize("method,path,body", WRITES)
def test_the_token_is_checked_before_the_origin_and_the_body(client, method, path, body):
    draft = make_draft(client)
    response = client.request(
        method,
        path.format(id=draft["id"]),
        content="not json",
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
    )
    assert response.status_code == 401
    response = client.request(
        method,
        path.format(id=draft["id"]),
        content="not json",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "text/plain"},
    )
    assert response.status_code == 403


def test_a_run_that_fails_after_its_timeout_keeps_the_timeout_error(tmp_path, caplog):
    release = threading.Event()
    runner = FakeRunner(RuntimeError(f"late {LLM_KEY}"), block=release)
    settings = make_settings(tmp_path, NEWSLETTER_RUN_TIMEOUT="0.1")
    with caplog.at_level(logging.WARNING):
        with TestClient(create_app(settings, runner=runner)) as client:
            run = run_and_wait(client)
            release.set()
            assert client.app.state.jobs.wait_idle(5)
            after = client.get(f"/api/runs/{run['id']}", headers=headers()).json()
    assert after == run
    assert after["error"] == "the run took longer than 0.1 seconds; its result is not used"
    assert "failed after its timeout" in caplog.text
    assert LLM_KEY not in caplog.text


# The store: ids, failed writes and stray files.


def test_a_draft_id_cannot_reach_another_json_file(tmp_path, client):
    from newsletter.api.store import DraftStore, NotFound

    run = run_and_wait(client)
    assert (tmp_path / "data" / "runs" / f"{run['id']}.json").is_file()
    store = DraftStore(tmp_path / "data")
    for draft_id in (f"../runs/{run['id']}", f"..%2Fruns%2F{run['id']}"):
        with pytest.raises(NotFound):
            store.get(draft_id)
    response = client.get(f"/api/drafts/..%2Fruns%2F{run['id']}", headers=headers())
    assert response.status_code == 404
    assert run["id"] not in response.text


def test_a_well_formed_id_with_no_file_is_not_found(client):
    missing = "0" * 32
    assert client.get(f"/api/drafts/{missing}", headers=headers()).status_code == 404
    assert client.get(f"/api/runs/{missing}", headers=headers()).status_code == 404
    response = client.post(f"/api/drafts/{missing}/approve", json={}, headers=headers())
    assert response.status_code == 404


@pytest.mark.parametrize("failing", ["json.dump", "os.replace"])
def test_a_failed_write_leaves_the_old_file_and_no_temp_file(tmp_path, monkeypatch, failing):
    from newsletter.api import store as store_module
    from newsletter.api.store import DraftStore

    drafts = DraftStore(tmp_path)
    draft = drafts.create(CONTENT, run_id="r")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    module, name = failing.split(".")
    monkeypatch.setattr(getattr(store_module, module), name, fail)
    with pytest.raises(OSError, match="disk full"):
        drafts.edit(draft["id"], "new text")
    monkeypatch.undo()

    assert sorted(p.name for p in (tmp_path / "drafts").iterdir()) == [f"{draft['id']}.json"]
    assert drafts.get(draft["id"])["text"] == TEXT


def test_stray_files_in_the_drafts_dir_are_skipped_and_logged(client, tmp_path, caplog):
    draft = make_draft(client)
    folder = tmp_path / "data" / "drafts"
    (folder / "notes.json").write_text('{"id": "notes"}', "utf-8")
    (folder / ("e" * 32 + ".json")).write_text("[1]", "utf-8")
    with caplog.at_level(logging.WARNING):
        listed = client.get("/api/drafts", headers=headers()).json()
    assert [d["id"] for d in listed] == [draft["id"]]
    assert "notes.json" in caplog.text
    assert "e" * 32 in caplog.text and "not an object" in caplog.text


def test_a_port_past_65535_is_refused():
    from newsletter.llm import ConfigError

    with pytest.raises(ConfigError, match="NEWSLETTER_API_PORT"):
        ApiSettings.from_env({"NEWSLETTER_API_PORT": "70000"})


# Live progress: every step of a run, polled while it goes.

LABELS = [
    "Fetching feeds",
    "Reading articles",
    "Grouping and ranking stories",
    "Writing claims",
    "Checking claims against sources",
    "Editing the issue",
    "Done",
]


class SteppedRunner:
    """The real pipeline with the recorded model and feeds, pausing after every progress
    event until the test has polled the run."""

    def __init__(self, llm, fetch=None, **changes):
        self.llm = llm
        self.fetch = fetch
        self.changes = changes
        self.emitted = threading.Semaphore(0)
        self.go = threading.Semaphore(0)

    def __call__(self, request, progress=None):
        from dataclasses import replace

        from newsletter.api.pipeline import run_issue
        from newsletter.voice import VoiceSettings
        from test_api_pipeline import web

        def report(*args, **kwargs):
            progress(*args, **kwargs)
            self.emitted.release()
            assert self.go.acquire(timeout=10)

        request = replace(request, **self.changes)
        fetch = self.fetch or web
        return run_issue(request, self.llm, fetch=fetch, voice=VoiceSettings(), progress=report)


def poll_every_step(
    tmp_path, llm, feeds=None, fetch=None, **changes
) -> tuple[list[list[dict]], dict]:
    from test_api_pipeline import CASE

    runner = SteppedRunner(llm, fetch, **changes)
    feeds = feeds or [CASE.feed_url]
    body = {"feeds": feeds, "title": "Open models week", "topic": "AI models"}
    seen = []
    with TestClient(create_app(make_settings(tmp_path), runner=runner)) as client:
        started = client.post("/api/runs", json=body, headers=headers())
        assert started.status_code == 202, started.text
        run_id = started.json()["id"]
        seen.append(started.json()["progress"])
        while not client.app.state.jobs.wait(run_id, 0):
            if runner.emitted.acquire(timeout=0.05):
                seen.append(client.get(f"/api/runs/{run_id}", headers=headers()).json()["progress"])
                runner.go.release()
        final = client.get(f"/api/runs/{run_id}", headers=headers()).json()
    seen.append(final["progress"])
    return seen, final


def states_of(seen: list[list[dict]], label: str) -> list[str]:
    states = []
    for progress in seen:
        (step,) = [step for step in progress if step["label"] == label]
        if not states or states[-1] != step["state"]:
            states.append(step["state"])
    return states


def test_a_polled_run_shows_every_step_go_waiting_running_done_in_order(tmp_path):
    from test_api_pipeline import replay

    seen, final = poll_every_step(tmp_path, replay())
    assert final["status"] == "done", final
    assert [step["label"] for step in final["progress"]] == LABELS
    assert all(step["state"] == "waiting" for step in seen[0])
    for label in LABELS:
        assert states_of(seen, label) == ["waiting", "running", "done"], label
    # One step at a time, in the order of the list.
    for progress in seen:
        running = [i for i, step in enumerate(progress) if step["state"] == "running"]
        assert len(running) <= 1
        if running:
            assert all(step["state"] == "done" for step in progress[: running[0]])
            assert all(step["state"] == "waiting" for step in progress[running[0] + 1 :])
    for step in final["progress"]:
        assert step["started_at"] and step["ended_at"]
        assert step["started_at"] <= step["ended_at"]
        assert set(step) == {
            "id",
            "label",
            "state",
            "done",
            "total",
            "detail",
            "started_at",
            "ended_at",
        }


def test_a_polled_run_whose_editor_fails_shows_the_editor_failed(tmp_path):
    from newsletter.llm import Settings
    from test_api_pipeline import CASE, Replay

    recorded = {k: v for k, v in CASE.recorded.items() if not k.startswith("edit")}
    seen, final = poll_every_step(tmp_path, Replay(Settings(models=CASE.models), recorded))
    assert final["status"] == "done"
    assert states_of(seen, "Editing the issue") == ["waiting", "running", "failed"]
    assert states_of(seen, "Done") == ["waiting", "running", "done"]


def test_a_polled_run_counts_each_stage_up_to_its_real_totals(tmp_path):
    from newsletter.llm import LLMClient, Settings
    from test_api_pipeline import GONE, mystery_gone
    from test_cli import ENV, NEWS, PAGE, by_meaning, model

    llm = LLMClient(Settings.from_env(ENV), transport=model(embed=by_meaning))
    seen, final = poll_every_step(
        tmp_path,
        llm,
        feeds=[NEWS, GONE],
        fetch=mystery_gone,
        pages=[PAGE],
        hackernews=HackerNewsConfig(min_score=100),
        topic="open-weight language models",
    )
    assert final["status"] == "done", final
    report = client_draft_report(tmp_path, final["draft_id"])
    calls = [step["step"] for step in report["steps"]]
    stories = sum(1 for call in calls if call.startswith("claims-"))
    checks = sum(1 for call in calls if call.startswith("check-"))
    steps = {step["id"]: step for step in final["progress"]}

    assert (steps["feeds"]["done"], steps["feeds"]["total"]) == (4, 4)
    assert steps["feeds"]["detail"] == f"3 of 4 loaded, 1 failed: {GONE} HTTP 404"
    assert (steps["articles"]["done"], steps["articles"]["total"]) == (2, 2)
    assert steps["articles"]["detail"] == (
        "1 of 2 fetched, 1 left out: the linked article could not be read"
    )
    assert (steps["claims"]["done"], steps["claims"]["total"]) == (stories, stories)
    assert (steps["check"]["done"], steps["check"]["total"]) == (checks, checks)

    # Polled as it went: every count only goes up, every total stays put.
    for step in ("feeds", "articles", "claims", "check"):
        counts = [
            (entry["done"], entry["total"])
            for progress in seen
            for entry in progress
            if entry["id"] == step and entry["done"] is not None
        ]
        assert len({done for done, _ in counts}) > 2, step
        assert [done for done, _ in counts] == sorted(done for done, _ in counts), step
        assert len({total for _, total in counts}) == 1, step


def client_draft_report(tmp_path, draft_id: str) -> dict:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        draft = client.get(f"/api/drafts/{draft_id}", headers=headers())
        assert draft.status_code == 200, draft.text
        return draft.json()["report"]


def test_no_model_text_or_key_appears_in_any_progress_detail(tmp_path):
    from test_api_pipeline import CASE, replay

    model_text = f"MODEL-WROTE-THIS {LLM_KEY}"  # The key rides in the model's reply.
    edit = {**CASE.recorded["edit"]["content"], "lead": model_text}
    llm = replay({"edit": {"content": edit}})
    seen, final = poll_every_step(tmp_path, llm)
    details = [step["detail"] for progress in seen for step in progress if step["detail"]]
    assert details
    for detail in details:
        assert "MODEL-WROTE-THIS" not in detail
        for secret in (TOKEN, LLM_KEY, VOICE_KEY):
            assert secret not in detail


class Emits:
    """A runner that reports the given progress events, then returns or raises."""

    def __init__(self, events, result=CONTENT):
        self.events = events
        self.result = result

    def __call__(self, request, progress=None):
        for step, state, extra in self.events:
            progress(step, state, **extra)
        if isinstance(self.result, Exception):
            raise self.result
        return dict(self.result)


def progress_of(tmp_path, runner, **changes) -> dict:
    with TestClient(create_app(make_settings(tmp_path, **changes), runner=runner)) as client:
        run = run_and_wait(client)
        assert client.app.state.jobs.wait_idle(5)
        stored = (tmp_path / "data" / "runs" / f"{run['id']}.json").read_text("utf-8")
    return run, {step["id"]: step for step in run["progress"]}, stored


def test_every_progress_detail_passes_through_the_redactor(tmp_path):
    leaky = f"3 of 4 feeds, key {LLM_KEY} {VOICE_KEY} {TOKEN}"
    events = [("feeds", "running", {}), ("feeds", "done", {"detail": leaky, "done": 3, "total": 4})]
    run, steps, stored = progress_of(tmp_path, Emits(events))
    assert steps["feeds"]["detail"] == "3 of 4 feeds, key [redacted] [redacted] [redacted]"
    assert (steps["feeds"]["done"], steps["feeds"]["total"]) == (3, 4)
    for secret in (TOKEN, LLM_KEY, VOICE_KEY):
        assert secret not in stored


def test_a_run_that_raises_fails_its_running_step_and_skips_the_rest(tmp_path):
    events = [("feeds", "running", {}), ("feeds", "done", {}), ("articles", "running", {})]
    run, steps, _ = progress_of(tmp_path, Emits(events, RuntimeError("boom")))
    assert run["status"] == "error"
    assert steps["feeds"]["state"] == "done"
    assert steps["articles"]["state"] == "failed" and steps["articles"]["ended_at"]
    assert {steps[i]["state"] for i in ("grouping", "claims", "check", "editor", "done")} == {
        "skipped"
    }


def test_a_run_past_its_timeout_fails_its_running_step_and_ignores_later_events(tmp_path):
    release = threading.Event()

    def slow(request, progress=None):
        progress("feeds", "running")
        release.wait(10)
        progress("feeds", "done", detail="late")
        return dict(CONTENT)

    settings = make_settings(tmp_path, NEWSLETTER_RUN_TIMEOUT="0.1")
    with TestClient(create_app(settings, runner=slow)) as client:
        run = run_and_wait(client)
        release.set()
        assert client.app.state.jobs.wait_idle(5)
        after = client.get(f"/api/runs/{run['id']}", headers=headers()).json()
    assert after == run
    steps = {step["id"]: step for step in after["progress"]}
    assert steps["feeds"]["state"] == "failed"
    assert steps["feeds"]["detail"] is None
    assert steps["done"]["state"] == "skipped"


def test_a_runner_that_reports_nothing_still_ends_with_done(tmp_path):
    run, steps, _ = progress_of(tmp_path, FakeRunner())
    assert run["status"] == "done"
    assert steps["done"]["state"] == "done"
    assert all(steps[i]["state"] == "skipped" for i in list(steps)[:-1])


def test_a_run_cut_off_by_a_restart_fails_its_running_step(tmp_path):
    release = threading.Event()
    settings = make_settings(tmp_path)

    def stuck(request, progress=None):
        progress("feeds", "running")
        release.wait(10)
        return dict(CONTENT)

    app = create_app(settings, runner=stuck)
    with TestClient(app) as client:
        response = client.post("/api/runs", json=RUN, headers=headers())
        run_id = response.json()["id"]
        for _ in range(100):
            got = client.get(f"/api/runs/{run_id}", headers=headers()).json()
            if got["progress"][0]["state"] == "running":
                break
            threading.Event().wait(0.02)
    with TestClient(create_app(settings, runner=FakeRunner())) as client:
        after = client.get(f"/api/runs/{run_id}", headers=headers()).json()
    release.set()
    assert after["status"] == "error"
    states = [step["state"] for step in after["progress"]]
    assert states == ["failed", *["skipped"] * 6]


def test_an_unknown_step_is_logged_and_changes_nothing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        run, steps, _ = progress_of(tmp_path, Emits([("nope", "running", {})]))
    assert run["status"] == "done"
    assert "no step 'nope' to report" in caplog.text
    assert [step["id"] for step in run["progress"]] == list(steps)
    assert len(steps) == 7


def test_a_second_running_event_keeps_when_the_step_started(tmp_path):
    def stored_claims():
        (path,) = (tmp_path / "data" / "runs").glob("*.json")
        return json.loads(path.read_text("utf-8"))["progress"][3]

    seen = []

    def twice(request, progress=None):
        progress("claims", "running", done=1, total=8)
        seen.append(stored_claims())
        threading.Event().wait(0.01)
        progress("claims", "running", done=3, total=8)
        seen.append(stored_claims())
        return dict(CONTENT)

    progress_of(tmp_path, twice)
    first, second = seen
    assert first["id"] == "claims"
    assert first["started_at"] and second["started_at"] == first["started_at"]
    assert first["ended_at"] is None and second["ended_at"] is None
    assert (second["done"], second["total"]) == (3, 8)


# A failed editor: the run says it in one plain sentence; the raw error is in the report.


def test_a_run_whose_editor_timed_out_shows_the_plain_sentence_and_no_raw_error(tmp_path):
    from newsletter.api.pipeline import run_issue
    from newsletter.llm import LLMClient, Settings
    from newsletter.voice import VoiceSettings
    from test_api_pipeline import EDITOR_TIMED_OUT, editor_answers, timed_out
    from test_cli import ENV, NEWS, run_web

    llm = LLMClient(Settings.from_env(ENV), transport=editor_answers(timed_out))

    def runner(request, progress):
        return run_issue(request, llm, fetch=run_web, voice=VoiceSettings(), progress=progress)

    with TestClient(create_app(make_settings(tmp_path), runner=runner)) as client:
        run = run_and_wait(client, {**RUN, "feeds": [NEWS]})
        draft = client.get(f"/api/drafts/{run['draft_id']}", headers=headers()).json()

    assert run["status"] == "done", run
    assert run["error"] is None
    assert run["editor_message"] == EDITOR_TIMED_OUT
    assert "timed out after" not in json.dumps(run)
    assert draft["report"]["editor_message"] == EDITOR_TIMED_OUT
    assert draft["report"]["editor_error"] == "edit: timed out after 300.0s"


def test_a_run_whose_editor_finished_has_no_editor_message(client):
    run = run_and_wait(client)
    assert run["status"] == "done"
    assert run["editor_message"] is None


def test_a_run_that_failed_has_no_editor_message(tmp_path):
    runner = FakeRunner(RuntimeError("boom"))
    with TestClient(create_app(make_settings(tmp_path), runner=runner)) as client:
        run = run_and_wait(client)
    assert run["status"] == "error"
    assert run["editor_message"] is None
