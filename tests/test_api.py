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

    def __call__(self, request):
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
