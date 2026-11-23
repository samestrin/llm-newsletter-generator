"""A run on the visitor's key or a demo code (epic 08.1 T2).

A visitor key goes only to its provider's base URL, a demo code only to the demo gateway,
and neither is in any response, stored file or log. A provider with no embedding model
(Synthetic) still runs: every item is its own story, in feed order, and the run says so.
Without the headers the run uses the server's own settings, as before.
"""

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from newsletter.api import ApiSettings, create_app
from newsletter.api.credentials import PROVIDERS, Credentials
from newsletter.api.pipeline import (
    NO_EMBEDDING,
    RunRequest,
    default_runner,
    run_issue,
    run_settings,
)
from newsletter.grouping import ungrouped_items
from newsletter.llm import ConfigError, LLMClient, Settings
from newsletter.voice import VoiceSettings
from test_api import ORIGIN, TOKEN, FakeRunner
from test_cli import ENV, NEWS, model, run_web
from test_grouping import make_item

KEY = "sk-visitor-marker-key-0123456789"
CODE = "demo-marker-code-0123456789"
SERVER_KEY = "sk-server-own-key-987"
GATEWAY = "http://gateway.test/v1"
DEMO_ENV = {
    "NEWSLETTER_DEMO_BASE_URL": GATEWAY,
    "NEWSLETTER_DEMO_FAST_MODEL": "gw-fast",
    "NEWSLETTER_DEMO_REASONING_MODEL": "gw-reasoning",
    "NEWSLETTER_DEMO_EMBEDDING_MODEL": "gw-embedding",
}
SYNTHETIC_NOTE = "Synthetic has no embedding model: stories are not grouped by meaning"


class Recorded:
    """The fake model endpoint, recording where each call went and with which bearer."""

    def __init__(self):
        self.calls: list[tuple[str, str | None, str]] = []
        self.inner = model()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((str(request.url), request.headers.get("authorization"), body["model"]))
        return self.inner.handle_request(request)


def env(tmp_path, **changes) -> dict[str, str]:
    """The server's settings: its own key and models, no voice key (no network)."""
    return {
        **ENV,
        "NEWSLETTER_API_TOKEN": TOKEN,
        "NEWSLETTER_DATA_DIR": str(tmp_path / "data"),
        "NEWSLETTER_API_KEY": SERVER_KEY,
        **changes,
    }


def run_with(tmp_path, caplog, credential_headers, *, answer=None, **env_changes):
    """Start one run through the API with the default runner on a fake transport and
    the fixture web, and return the run, its draft, every response body and the
    transport's calls. `answer` replaces the fake model endpoint's replies."""
    settings = ApiSettings.from_env(env(tmp_path, **env_changes))
    transport = Recorded()
    if answer is not None:
        transport.inner = httpx.MockTransport(answer)
    runner = default_runner(settings.env, transport=httpx.MockTransport(transport), fetch=run_web)
    sent = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN, **credential_headers}
    bodies = []
    with caplog.at_level(logging.DEBUG):
        with TestClient(create_app(settings, runner=runner)) as client:
            started = client.post("/api/runs", json={"feeds": [NEWS], "topic": "AI"}, headers=sent)
            assert started.status_code == 202, started.text
            bodies.append(started.text)
            assert client.app.state.jobs.wait(started.json()["id"], 30)
            run = client.get(f"/api/runs/{started.json()['id']}", headers=sent)
            bodies.append(run.text)
            draft = None
            if run.json()["draft_id"]:
                draft = client.get(f"/api/drafts/{run.json()['draft_id']}", headers=sent)
                bodies.append(draft.text)
                bodies.append(client.get("/api/drafts", headers=sent).text)
                draft = draft.json()
    return run.json(), draft, bodies, transport.calls


def stored_text(tmp_path) -> str:
    return "\n".join(
        path.read_text("utf-8") for path in (tmp_path / "data").rglob("*") if path.is_file()
    )


def assert_nowhere(secret: str, tmp_path, caplog, bodies) -> None:
    for body in bodies:
        assert secret not in body
    assert secret not in stored_text(tmp_path)
    assert secret not in caplog.text


# A visitor key.


def test_a_visitor_key_goes_only_to_its_providers_base_url(tmp_path, caplog):
    run, draft, bodies, calls = run_with(
        tmp_path, caplog, {"X-Model-Preset": "openrouter", "X-Model-Key": KEY}
    )

    assert run["status"] == "done", run
    assert calls
    openrouter = PROVIDERS["openrouter"]
    for url, bearer, _ in calls:
        assert url.startswith(openrouter.base_url + "/")
        assert bearer == f"Bearer {KEY}"
    used = {model_name for _, _, model_name in calls}
    assert used == {
        openrouter.fast_model,
        openrouter.reasoning_model,
        openrouter.embedding_model,
    }
    assert draft["report"]["grouping_note"] is None
    assert_nowhere(KEY, tmp_path, caplog, bodies)
    assert_nowhere(SERVER_KEY, tmp_path, caplog, bodies)


def test_a_visitor_key_echoed_by_the_provider_is_redacted_everywhere(tmp_path, caplog):
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {request.headers['authorization']}")

    run, draft, bodies, calls = run_with(
        tmp_path, caplog, {"X-Model-Preset": "openai", "X-Model-Key": KEY}, answer=echo
    )

    # A failed model call is kept in the report, not raised: the run itself finishes.
    assert run["status"] == "done", run
    assert calls
    failed = [step["error"] for step in draft["report"]["steps"] if not step["ok"]]
    assert failed and all("bad key Bearer [redacted]" in error for error in failed)
    assert_nowhere(KEY, tmp_path, caplog, bodies)


def test_a_run_error_holding_the_visitor_key_is_redacted_by_jobs(tmp_path, caplog):
    """The runner's own error, not the client's: `Jobs` redacts the run's own key."""

    class Leaky(FakeRunner):
        def __call__(self, request, progress=None):
            self.requests.append(request)
            if progress is not None:
                progress("feeds", "running", detail=f"reading with {request.credentials.secret}")
            raise RuntimeError(f"boom {request.credentials.secret} and {TOKEN}")

    runner = Leaky()
    settings = ApiSettings.from_env(env(tmp_path))
    sent = {
        "Authorization": f"Bearer {TOKEN}",
        "Origin": ORIGIN,
        "X-Model-Preset": "openai",
        "X-Model-Key": KEY,
    }
    with caplog.at_level(logging.DEBUG):
        with TestClient(create_app(settings, runner=runner)) as client:
            started = client.post("/api/runs", json={"feeds": [NEWS]}, headers=sent)
            assert client.app.state.jobs.wait(started.json()["id"], 5)
            run = client.get(f"/api/runs/{started.json()['id']}", headers=sent)
    assert runner.requests[0].credentials.secret == KEY
    assert run.json()["status"] == "error"
    assert run.json()["error"] == "RuntimeError: boom [redacted] and [redacted]"
    feeds = next(step for step in run.json()["progress"] if step["id"] == "feeds")
    assert feeds["detail"] == "reading with [redacted]"
    assert_nowhere(KEY, tmp_path, caplog, [started.text, run.text])
    assert_nowhere(TOKEN, tmp_path, caplog, [started.text, run.text])


def test_one_runs_key_is_not_redacted_from_the_next_run(tmp_path):
    """The per-run list is the run's own: the app's redactor is not widened for good."""
    runner = FakeRunner({"title": "T", "issue_date": "2026-11-02", "text": KEY, "report": {}})
    settings = ApiSettings.from_env(env(tmp_path))
    base = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN}
    with TestClient(create_app(settings, runner=runner)) as client:
        keyed = {**base, "X-Model-Preset": "openai", "X-Model-Key": KEY}
        first = client.post("/api/runs", json={"feeds": [NEWS]}, headers=keyed).json()
        assert client.app.state.jobs.wait(first["id"], 5)
        second = client.post("/api/runs", json={"feeds": [NEWS]}, headers=base).json()
        assert client.app.state.jobs.wait(second["id"], 5)
        texts = [
            client.get(
                f"/api/drafts/{client.get(f'/api/runs/{r}', headers=base).json()['draft_id']}",
                headers=base,
            ).json()["text"]
            for r in (first["id"], second["id"])
        ]
    assert texts == ["[redacted]", KEY]


def test_a_visitor_key_builds_the_runs_settings_from_its_provider_only():
    server = Settings.from_env(
        {
            **ENV,
            "NEWSLETTER_API_KEY": SERVER_KEY,
            "NEWSLETTER_FAST_PRICE_INPUT": "1",
            "NEWSLETTER_TIMEOUT": "7",
        }
    )
    settings = run_settings(
        {
            **ENV,
            "NEWSLETTER_API_KEY": SERVER_KEY,
            "NEWSLETTER_FAST_PRICE_INPUT": "1",
            "NEWSLETTER_TIMEOUT": "7",
        },
        Credentials(PROVIDERS["synthetic"], KEY),
    )
    assert settings.base_url == PROVIDERS["synthetic"].base_url
    assert settings.api_key == KEY
    assert settings.models == {
        "fast": PROVIDERS["synthetic"].fast_model,
        "reasoning": PROVIDERS["synthetic"].reasoning_model,
    }
    # The server's prices are for its own models; a visitor's provider reports its own.
    assert settings.prices == {}
    assert settings.timeout == server.timeout == 7
    assert settings.max_retries == server.max_retries
    assert KEY not in repr(settings)


# A demo code.


def test_a_demo_code_goes_only_to_the_demo_gateway(tmp_path, caplog):
    run, draft, bodies, calls = run_with(
        tmp_path, caplog, {"X-Model-Preset": "demo", "X-Demo-Code": CODE}, **DEMO_ENV
    )

    assert run["status"] == "done", run
    assert calls
    for url, bearer, _ in calls:
        assert url.startswith(GATEWAY + "/")
        assert bearer == f"Bearer {CODE}"
    assert {name for _, _, name in calls} == {"gw-fast", "gw-reasoning", "gw-embedding"}
    assert_nowhere(CODE, tmp_path, caplog, bodies)
    assert_nowhere(SERVER_KEY, tmp_path, caplog, bodies)


def test_a_demo_code_alone_picks_the_demo_provider(tmp_path, caplog):
    run, _, _, calls = run_with(tmp_path, caplog, {"X-Demo-Code": CODE}, **DEMO_ENV)
    assert run["status"] == "done", run
    assert {url.split("/v1/")[0] for url, _, _ in calls} == {"http://gateway.test"}


def test_demo_models_left_unset_are_the_providers_demo_names():
    settings = run_settings(
        {"NEWSLETTER_DEMO_BASE_URL": GATEWAY + "/"}, Credentials(PROVIDERS["demo"], CODE)
    )
    assert settings.base_url == GATEWAY
    assert settings.models == {
        "fast": "demo-fast",
        "reasoning": "demo-reasoning",
        "embedding": "demo-embedding",
    }


def test_a_demo_code_without_a_gateway_is_refused_before_the_run(tmp_path):
    runner = FakeRunner()
    settings = ApiSettings.from_env(env(tmp_path))
    sent = {
        "Authorization": f"Bearer {TOKEN}",
        "Origin": ORIGIN,
        "X-Model-Preset": "demo",
        "X-Demo-Code": CODE,
    }
    with TestClient(create_app(settings, runner=runner)) as client:
        response = client.post("/api/runs", json={"feeds": [NEWS]}, headers=sent)
    assert response.status_code == 503
    assert "NEWSLETTER_DEMO_BASE_URL" in response.json()["detail"]
    assert CODE not in response.text
    assert runner.requests == []


def test_run_settings_names_the_missing_gateway():
    with pytest.raises(ConfigError, match="NEWSLETTER_DEMO_BASE_URL"):
        run_settings({}, Credentials(PROVIDERS["demo"], CODE))


# A provider with no embedding model.


def test_a_synthetic_run_completes_with_ungrouped_stories_and_the_plain_note(tmp_path, caplog):
    run, draft, bodies, calls = run_with(
        tmp_path, caplog, {"X-Model-Preset": "synthetic", "X-Model-Key": KEY}
    )

    assert run["status"] == "done", run
    assert not any(url.endswith("/embeddings") for url, _, _ in calls)
    grouping = next(step for step in run["progress"] if step["id"] == "grouping")
    assert grouping["state"] == "done"
    assert SYNTHETIC_NOTE in grouping["detail"]
    report = draft["report"]
    assert report["grouping_note"] == SYNTHETIC_NOTE
    assert report["grouping_error"] is None
    assert SYNTHETIC_NOTE in draft["text"]
    assert_nowhere(KEY, tmp_path, caplog, bodies)


def test_without_an_embedding_model_each_item_is_its_own_story_in_feed_order():
    llm = LLMClient(run_settings(ENV, Credentials(PROVIDERS["synthetic"], KEY)), transport=model())
    request = RunRequest([NEWS], topic="AI", credentials=Credentials(PROVIDERS["synthetic"], KEY))
    content = run_issue(request, llm, fetch=run_web, voice=VoiceSettings())

    report = content["report"]
    assert report["grouping_note"] == NO_EMBEDDING.format(label="Synthetic") == SYNTHETIC_NOTE
    assert report["left_out"] == []
    assert not any(step["tier"] == "embedding" for step in report["steps"])
    # One story per item, numbered in feed order: s1 is the feed's first item.
    claims = [step["step"] for step in report["steps"] if step["step"].startswith("claims-")]
    assert claims == ["claims-s1", "claims-s2"]


def test_ungrouped_items_keep_feed_order_and_list_only_items_with_no_text():
    items = [make_item("First", summary="a"), make_item("", summary=""), make_item("Second")]
    grouping = ungrouped_items(items, note="no embedding model")

    assert [[item.title for item in story] for story in grouping.stories.values()] == [
        ["First"],
        ["Second"],
    ]
    assert list(grouping.stories) == ["s1", "s2"]
    assert grouping.scores == {"s1": None, "s2": None}
    assert grouping.note == "no embedding model"
    assert grouping.error is None
    assert [entry.reason for entry in grouping.left_out] == [
        "dropped: no title or summary to compare"
    ]


def test_the_note_alone_does_not_make_the_report_not_ok():
    from newsletter.render import RunReport

    assert RunReport(grouping_note=SYNTHETIC_NOTE).ok is True


# Server-key mode.


def test_without_headers_the_run_uses_the_servers_own_settings(tmp_path, caplog):
    run, draft, _, calls = run_with(tmp_path, caplog, {})

    assert run["status"] == "done", run
    for url, bearer, _ in calls:
        assert url.startswith(ENV["NEWSLETTER_BASE_URL"] + "/")
        assert bearer == f"Bearer {SERVER_KEY}"
    assert {name for _, _, name in calls} == {"fast-1", "reason-1", "embed-1"}
    assert draft["report"]["grouping_note"] is None


def test_run_settings_without_credentials_are_the_servers():
    assert run_settings(ENV, None) == Settings.from_env(ENV)


def test_the_token_is_checked_before_the_credential_headers(tmp_path):
    runner = FakeRunner()
    settings = ApiSettings.from_env(env(tmp_path))
    sent = {"Authorization": "Bearer wrong", "Origin": ORIGIN, "X-Model-Key": KEY}
    with TestClient(create_app(settings, runner=runner)) as client:
        response = client.post("/api/runs", json={"feeds": [NEWS]}, headers=sent)
    assert response.status_code == 401
    assert KEY not in response.text
    assert runner.requests == []
