"""Drafts belong to the browser that made them, in demo mode.

`POST /api/runs` issues an owner token once; only its sha256 is stored, as
`owner_sha256` on the run and its draft. Sent back as `X-Owner-Token`, it scopes the draft
and run routes: a draft with no owner (the seeded example) is readable by everyone and
writable by no one, any other draft answers only to its own token and is a 404 to the
rest. With demo mode off every route behaves as before.
"""

import hashlib
import json
import logging
import secrets

import pytest
from fastapi.testclient import TestClient

from newsletter.api import ApiSettings, create_app
from newsletter.api.store import DraftStore, RunStore, new_id, now
from test_api import CONTENT, ORIGIN, TOKEN, FakeRunner, headers, make_settings, sha
from test_feed_check import GOOD, Web, public

RUN = {"feeds": [GOOD], "title": "Weekly"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def demo(tmp_path, *, runner=None, demo_mode=True) -> TestClient:
    env = {"NEWSLETTER_DEMO": "1"} if demo_mode else {}
    settings = make_settings(tmp_path, **env)
    return TestClient(
        create_app(settings, runner=runner or FakeRunner(), fetch=Web(), resolve=public)
    )


def seed_example(tmp_path) -> dict:
    """A draft with no owner and its run, as `seed.seed_sample` writes them."""
    data = tmp_path / "data"
    run = {"id": new_id(), "status": "done", "error": None, "draft_id": None, "feeds": []}
    run.update(title="Example", created_at=now(), started_at=None, finished_at=None)
    draft = DraftStore(data).create(CONTENT, run_id=run["id"])
    run["draft_id"] = draft["id"]
    RunStore(data).save(run)
    return draft


def owner(token: str | None) -> dict[str, str]:
    return headers(**({"X-Owner-Token": token} if token is not None else {}))


def start(client) -> tuple[dict, str]:
    """Run once; the finished run and the owner token its start returned."""
    response = client.post("/api/runs", json=RUN, headers=headers())
    assert response.status_code == 202, response.text
    started = response.json()
    assert client.app.state.jobs.wait(started["id"], 5)
    token = started["owner_token"]
    run = client.get(f"/api/runs/{started['id']}", headers=owner(token))
    assert run.status_code == 200, run.text
    return run.json(), token


def listed(client, token) -> set[str]:
    response = client.get("/api/drafts", headers=owner(token))
    assert response.status_code == 200
    return {draft["id"] for draft in response.json()}


def files(tmp_path) -> str:
    return "\n".join(p.read_text("utf-8") for p in (tmp_path / "data").rglob("*.json"))


# The token.


def test_a_demo_run_returns_an_owner_token_and_stores_only_its_sha256(tmp_path):
    with demo(tmp_path) as client:
        run, token = start(client)
        again, other = start(client)
    assert len(token) >= 22 and len(token.encode()) * 6 // 8 >= 16  # url-safe base64
    assert len(set(token)) > 8
    assert token != other
    stored_run = json.loads((tmp_path / "data" / "runs" / f"{run['id']}.json").read_text())
    stored_draft = json.loads(
        (tmp_path / "data" / "drafts" / f"{run['draft_id']}.json").read_text()
    )
    assert stored_run["owner_sha256"] == sha256(token)
    assert stored_draft["owner_sha256"] == sha256(token)
    assert "owner_token" not in stored_run


def test_the_token_is_returned_once_and_no_view_shows_it_or_its_hash(tmp_path):
    with demo(tmp_path) as client:
        run, token = start(client)
        draft_id = run["draft_id"]
        bodies = [
            client.get(f"/api/runs/{run['id']}", headers=owner(token)).text,
            client.get("/api/drafts", headers=owner(token)).text,
            client.get(f"/api/drafts/{draft_id}", headers=owner(token)).text,
            client.put(f"/api/drafts/{draft_id}", json={"text": "x"}, headers=owner(token)).text,
            client.post(f"/api/drafts/{draft_id}/approve", json={}, headers=owner(token)).text,
            client.get(f"/api/drafts/{draft_id}/email", headers=owner(token)).text,
        ]
    for body in bodies:
        assert token not in body
        assert sha256(token) not in body
        assert "owner_sha256" not in body
        assert "owner_token" not in body


def test_no_token_is_in_the_stored_files_or_a_captured_log(tmp_path, caplog):
    runner = FakeRunner()
    with caplog.at_level(logging.DEBUG), demo(tmp_path, runner=runner) as client:
        run, token = start(client)
        draft_id = run["draft_id"]
        client.put(f"/api/drafts/{draft_id}", json={"text": "edited"}, headers=owner(token))
        client.post(f"/api/drafts/{draft_id}/approve", json={}, headers=owner(token))
        client.get(f"/api/drafts/{draft_id}/email", headers=owner(token))
        client.get(f"/api/drafts/{new_id()}", headers=owner(token))
    assert token not in files(tmp_path)
    assert sha256(token) in files(tmp_path)  # the marker test can see what it looks for
    assert token not in caplog.text
    assert token not in repr(runner.requests)


# Two browsers, A and B.


def test_each_list_holds_the_example_and_its_own_drafts_only(tmp_path):
    example = seed_example(tmp_path)
    with demo(tmp_path) as client:
        a_run, a = start(client)
        b_run, b = start(client)
        assert listed(client, a) == {example["id"], a_run["draft_id"]}
        assert listed(client, b) == {example["id"], b_run["draft_id"]}
        assert listed(client, None) == {example["id"]}


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "/api/drafts/{id}", None),
        ("put", "/api/drafts/{id}", {"text": "taken over"}),
        ("post", "/api/drafts/{id}/approve", {}),
        ("get", "/api/drafts/{id}/email", None),
    ],
)
@pytest.mark.parametrize("who", ["a", "wrong", "none"])
def test_another_browsers_draft_is_not_found(tmp_path, method, path, body, who):
    with demo(tmp_path) as client:
        a_run, a = start(client)
        b_run, b = start(client)
        b_id = b_run["draft_id"]
        client.post(f"/api/drafts/{b_id}/approve", json={}, headers=owner(b))
        before = client.get(f"/api/drafts/{b_id}", headers=owner(b)).json()
        sent = {"a": a, "wrong": secrets.token_urlsafe(32), "none": None}[who]
        response = client.request(method, path.format(id=b_id), json=body, headers=owner(sent))
        missing = client.request(method, path.format(id=new_id()), json=body, headers=owner(a))
        after = client.get(f"/api/drafts/{b_id}", headers=owner(b)).json()
    assert response.status_code == 404
    assert response.json() == missing.json()  # the same answer as no draft at all
    assert after == before


@pytest.mark.parametrize("who", ["a", "wrong", "none"])
def test_another_browsers_run_is_not_found(tmp_path, who):
    with demo(tmp_path) as client:
        a_run, a = start(client)
        b_run, b = start(client)
        sent = {"a": a, "wrong": secrets.token_urlsafe(32), "none": None}[who]
        response = client.get(f"/api/runs/{b_run['id']}", headers=owner(sent))
    assert response.status_code == 404


def test_a_browser_reads_edits_approves_and_emails_its_own_draft(tmp_path):
    with demo(tmp_path) as client:
        run, token = start(client)
        draft_id = run["draft_id"]
        read = client.get(f"/api/drafts/{draft_id}", headers=owner(token))
        edit = client.put(f"/api/drafts/{draft_id}", json={"text": "# Mine\n"}, headers=owner(token))
        approve = client.post(
            f"/api/drafts/{draft_id}/approve", json={"sha256": sha("# Mine\n")}, headers=owner(token)
        )
        email = client.get(f"/api/drafts/{draft_id}/email", headers=owner(token))
    assert read.status_code == 200
    assert edit.status_code == 200
    assert approve.status_code == 200 and approve.json()["approved"] is True
    assert email.status_code == 200 and "Mine" in email.json()["html"]


# The example: readable by everyone, changed by no one.


@pytest.mark.parametrize("who", ["a", "none"])
def test_everyone_reads_the_example_and_its_run(tmp_path, who):
    example = seed_example(tmp_path)
    with demo(tmp_path) as client:
        _, a = start(client)
        sent = a if who == "a" else None
        draft = client.get(f"/api/drafts/{example['id']}", headers=owner(sent))
        run = client.get(f"/api/runs/{example['run_id']}", headers=owner(sent))
        email = client.get(f"/api/drafts/{example['id']}/email", headers=owner(sent))
    assert draft.status_code == 200
    assert draft.json()["text"] == CONTENT["text"]
    assert run.status_code == 200
    assert email.status_code == 409  # readable; not approved, and no one can approve it


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("put", "/api/drafts/{id}", {"text": "vandalised"}),
        ("post", "/api/drafts/{id}/approve", {}),
    ],
)
@pytest.mark.parametrize("who", ["a", "none"])
def test_no_one_edits_or_approves_the_example(tmp_path, method, path, body, who):
    example = seed_example(tmp_path)
    with demo(tmp_path) as client:
        _, a = start(client)
        sent = a if who == "a" else None
        response = client.request(method, path.format(id=example["id"]), json=body, headers=owner(sent))
        after = client.get(f"/api/drafts/{example['id']}", headers=owner(None)).json()
    assert response.status_code == 403
    assert "example" in response.json()["detail"]
    assert after["text"] == CONTENT["text"]
    assert after["approved"] is False


def test_the_seeded_sample_has_no_owner(tmp_path):
    env = {"NEWSLETTER_API_TOKEN": TOKEN, "NEWSLETTER_DATA_DIR": str(tmp_path / "data")}
    settings = ApiSettings.from_env({**env, "NEWSLETTER_DEMO": "1"})
    with TestClient(create_app(settings, seed_sample=True)) as client:
        (draft,) = client.get("/api/drafts", headers=headers()).json()
    stored = json.loads((tmp_path / "data" / "drafts" / f"{draft['id']}.json").read_text())
    assert not stored.get("owner_sha256")


# Demo mode off: the server-key review host, as before.


def test_with_demo_mode_off_no_token_is_issued_or_stored(tmp_path):
    with demo(tmp_path, demo_mode=False) as client:
        response = client.post("/api/runs", json=RUN, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
    assert "owner_token" not in response.json()
    assert "owner_sha256" not in files(tmp_path)


def test_with_demo_mode_off_every_draft_is_open_whatever_the_header(tmp_path):
    example = seed_example(tmp_path)
    with demo(tmp_path, demo_mode=False) as client:
        response = client.post("/api/runs", json=RUN, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
        run = client.get(f"/api/runs/{response.json()['id']}", headers=owner("anything"))
        draft_id = run.json()["draft_id"]
        assert listed(client, "anything") == {example["id"], draft_id}
        edit = client.put(f"/api/drafts/{example['id']}", json={"text": "ok"}, headers=headers())
        approve = client.post(f"/api/drafts/{draft_id}/approve", json={}, headers=owner("x"))
    assert run.status_code == 200
    assert edit.status_code == 200
    assert approve.status_code == 200


def test_a_draft_made_in_demo_mode_stays_open_once_demo_mode_is_off(tmp_path):
    with demo(tmp_path) as client:
        run, _ = start(client)
    with demo(tmp_path, demo_mode=False) as client:
        response = client.get(f"/api/drafts/{run['draft_id']}", headers=headers())
    assert response.status_code == 200


# The browser and the spec.


def test_the_page_may_send_the_owner_token_header(tmp_path):
    with demo(tmp_path) as client:
        response = client.options(
            "/api/drafts",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,x-owner-token",
            },
        )
    assert response.status_code == 200
    assert "x-owner-token" in response.headers["access-control-allow-headers"].lower()


SCOPED = [
    ("get", "/api/runs/{run_id}"),
    ("get", "/api/drafts"),
    ("get", "/api/drafts/{draft_id}"),
    ("put", "/api/drafts/{draft_id}"),
    ("post", "/api/drafts/{draft_id}/approve"),
    ("get", "/api/drafts/{draft_id}/email"),
]


def test_the_header_and_the_owner_token_field_are_in_the_spec(tmp_path):
    with demo(tmp_path) as client:
        spec = client.get("/api/openapi.json").json()
    for method, path in SCOPED:
        parameters = spec["paths"][path][method].get("parameters", [])
        names = {(p["in"], p["name"].lower()) for p in parameters}
        assert ("header", "x-owner-token") in names, (method, path)
    assert "owner_token" in spec["components"]["schemas"]["RunView"]["properties"]
    for method, path in (("put", "/api/drafts/{draft_id}"), ("post", "/api/drafts/{draft_id}/approve")):
        assert "example" in spec["paths"][path][method]["responses"]["403"]["description"]
