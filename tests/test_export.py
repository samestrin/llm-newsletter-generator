import json
import logging

import httpx
import pytest

from newsletter.api.store import DraftStore
from newsletter.archive import ApprovalError, demo_approval, human_approval
from newsletter.export import (
    Beehiiv,
    Buttondown,
    Resend,
    connectors_from_env,
    export_draft,
    export_stored_draft,
)
from newsletter.llm import ConfigError

TEXT = "# AI this week\n\nA lab shipped a <small> model.\n\nSee https://lab.example.org/m."
TITLE = "AI this week"
KEYS = {
    "buttondown": "bd-secret-key-111",
    "beehiiv": "bh-secret-key-222",
    "resend": "re_secret-key-333",
}
ENV = {
    "BUTTONDOWN_API_KEY": KEYS["buttondown"],
    "BEEHIIV_API_KEY": KEYS["beehiiv"],
    "BEEHIIV_PUBLICATION_ID": "pub_123",
    "RESEND_API_KEY": KEYS["resend"],
    "RESEND_SEGMENT_ID": "seg_456",
    "RESEND_FROM": "Sam <sam@example.org>",
}


class FakePlatforms:
    """Buttondown, Beehiiv and Resend in one transport, answering each create call.

    `fail` maps a host to a reply that replaces the normal one: an int is sent as that
    HTTP status with the caller's key echoed in the body, an exception is raised as the
    transport error, a dict is sent as the JSON body with status 200.
    """

    def __init__(self, fail=None):
        self.fail = fail or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.fail.get(request.url.host)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            echoed = request.headers.get("authorization", "")
            return httpx.Response(reply, json={"detail": f"rejected {echoed}"})
        if isinstance(reply, dict):
            return httpx.Response(200, json=reply)
        body = json.loads(request.content)
        if request.url.host == "api.buttondown.com":
            return httpx.Response(201, json={"id": "em_1", "status": body.get("status")})
        if request.url.host == "api.beehiiv.com":
            return httpx.Response(201, json={"data": {"id": "post_1", "status": "draft"}})
        return httpx.Response(201, json={"object": "broadcast", "id": "bc_1"})

    def bodies(self) -> dict[str, dict]:
        return {r.url.host: json.loads(r.content) for r in self.requests}


def all_connectors():
    return connectors_from_env(ENV)


def approved():
    return human_approval(TEXT)


# --- the approval gate ---------------------------------------------------------------


def test_export_without_approval_is_refused_and_nothing_is_sent():
    platforms = FakePlatforms()
    with pytest.raises(ApprovalError, match="not approved"):
        export_draft(
            TEXT,
            title=TITLE,
            approval=None,
            connectors=all_connectors(),
            transport=httpx.MockTransport(platforms),
        )
    assert platforms.requests == []


def test_export_after_an_edit_is_refused_and_nothing_is_sent():
    platforms = FakePlatforms()
    with pytest.raises(ApprovalError, match="changed"):
        export_draft(
            TEXT + " edited",
            title=TITLE,
            approval=approved(),
            connectors=all_connectors(),
            transport=httpx.MockTransport(platforms),
        )
    assert platforms.requests == []


def test_an_automated_demo_approval_is_refused():
    platforms = FakePlatforms()
    with pytest.raises(ApprovalError, match="archive pages only"):
        export_draft(
            TEXT,
            title=TITLE,
            approval=demo_approval(),
            connectors=all_connectors(),
            transport=httpx.MockTransport(platforms),
        )
    assert platforms.requests == []


def test_a_stored_draft_edited_after_approval_is_refused(tmp_path):
    store = DraftStore(tmp_path)
    draft = store.create({"title": TITLE, "issue_date": "2026-11-03", "text": TEXT}, run_id="r")
    store.approve(draft["id"])
    edited = store.edit(draft["id"], TEXT + " edited")
    platforms = FakePlatforms()
    with pytest.raises(ApprovalError, match="not approved"):
        export_stored_draft(
            edited, connectors=all_connectors(), transport=httpx.MockTransport(platforms)
        )
    assert platforms.requests == []


def test_a_stored_draft_whose_text_no_longer_matches_its_approval_is_refused(tmp_path):
    store = DraftStore(tmp_path)
    draft = store.create({"title": TITLE, "issue_date": "2026-11-03", "text": TEXT}, run_id="r")
    approved_draft = store.approve(draft["id"])
    tampered = {**approved_draft, "text": TEXT + " changed on disk"}
    platforms = FakePlatforms()
    with pytest.raises(ApprovalError, match="changed"):
        export_stored_draft(
            tampered, connectors=all_connectors(), transport=httpx.MockTransport(platforms)
        )
    assert platforms.requests == []


def test_an_approved_stored_draft_exports(tmp_path):
    store = DraftStore(tmp_path)
    draft = store.create({"title": TITLE, "issue_date": "2026-11-03", "text": TEXT}, run_id="r")
    report = export_stored_draft(
        store.approve(draft["id"]),
        connectors=all_connectors(),
        transport=httpx.MockTransport(FakePlatforms()),
    )
    assert report.ok
    assert [r.connector for r in report.exported] == ["buttondown", "beehiiv", "resend"]


# --- draft status, never a send call ----------------------------------------------------


def test_each_connector_creates_a_draft_and_never_sends():
    platforms = FakePlatforms()
    report = export_draft(
        TEXT,
        title=TITLE,
        approval=approved(),
        connectors=all_connectors(),
        transport=httpx.MockTransport(platforms),
    )

    assert report.ok
    assert [(r.connector, r.draft_id) for r in report.exported] == [
        ("buttondown", "em_1"),
        ("beehiiv", "post_1"),
        ("resend", "bc_1"),
    ]
    calls = [(r.method, str(r.url)) for r in platforms.requests]
    assert calls == [
        ("POST", "https://api.buttondown.com/v1/emails"),
        ("POST", "https://api.beehiiv.com/v2/publications/pub_123/posts"),
        ("POST", "https://api.resend.com/broadcasts"),
    ]
    assert not any("send" in r.url.path for r in platforms.requests)

    bodies = platforms.bodies()
    buttondown = bodies["api.buttondown.com"]
    assert buttondown == {"subject": TITLE, "body": TEXT, "status": "draft"}

    beehiiv = bodies["api.beehiiv.com"]
    assert beehiiv["status"] == "draft"
    assert beehiiv["title"] == TITLE
    assert "scheduled_at" not in beehiiv
    assert "&lt;small&gt;" in beehiiv["body_content"]

    resend = bodies["api.resend.com"]
    assert resend["send"] is False
    assert "scheduled_at" not in resend
    assert resend["segment_id"] == "seg_456"
    assert resend["from"] == "Sam <sam@example.org>"
    assert resend["subject"] == TITLE
    assert resend["text"] == TEXT
    assert "<small>" not in resend["html"]


def test_each_connector_sends_its_own_key_in_its_own_header():
    platforms = FakePlatforms()
    export_draft(
        TEXT,
        title=TITLE,
        approval=approved(),
        connectors=all_connectors(),
        transport=httpx.MockTransport(platforms),
    )
    auth = {r.url.host: r.headers["authorization"] for r in platforms.requests}
    assert auth == {
        "api.buttondown.com": f"Token {KEYS['buttondown']}",
        "api.beehiiv.com": f"Bearer {KEYS['beehiiv']}",
        "api.resend.com": f"Bearer {KEYS['resend']}",
    }


def test_a_platform_answering_with_a_status_other_than_draft_is_a_failure():
    platforms = FakePlatforms(fail={"api.buttondown.com": {"id": "em_1", "status": "sent"}})
    report = export_draft(
        TEXT,
        title=TITLE,
        approval=approved(),
        connectors=all_connectors(),
        transport=httpx.MockTransport(platforms),
    )
    assert not report.ok
    [failed] = report.failed
    assert failed.connector == "buttondown"
    assert "sent" in failed.error and "draft" in failed.error


def test_a_connector_with_no_draft_only_endpoint_is_dropped_and_listed():
    class SendsOnly(Buttondown):
        name = "sends-only"
        draft_only = False

    platforms = FakePlatforms()
    connectors = [SendsOnly(api_key="k-sends-only"), *all_connectors()]
    report = export_draft(
        TEXT,
        title=TITLE,
        approval=approved(),
        connectors=connectors,
        transport=httpx.MockTransport(platforms),
    )
    assert report.dropped == ["sends-only"]
    assert len(platforms.requests) == 3
    assert "sends-only" in report.summary()
    assert "no draft-only endpoint" in report.summary()


# --- failures are reported, never shown as success ----------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [
        (500, "HTTP 500"),
        (401, "HTTP 401"),
        (httpx.ReadTimeout("slow"), "timed out after 7s"),
        (httpx.ConnectError("refused"), "request failed"),
        ({"no": "id"}, "no draft id"),
    ],
)
def test_a_failing_api_ends_in_a_reported_error(reply, expected):
    platforms = FakePlatforms(fail={"api.beehiiv.com": reply})
    connectors = connectors_from_env({**ENV, "NEWSLETTER_EXPORT_TIMEOUT": "7"})
    report = export_draft(
        TEXT,
        title=TITLE,
        approval=approved(),
        connectors=connectors,
        transport=httpx.MockTransport(platforms),
    )

    assert not report.ok
    [failed] = report.failed
    assert failed.connector == "beehiiv"
    assert not failed.ok
    assert failed.draft_id is None
    assert expected in failed.error
    # The other connectors still ran, and the summary leads with the failure.
    assert [r.connector for r in report.exported] == ["buttondown", "resend"]
    summary = report.summary()
    assert summary.startswith("Export failed")
    assert "beehiiv" in summary and expected in summary


def test_a_report_with_nothing_configured_is_not_a_success():
    report = export_draft(TEXT, title=TITLE, approval=approved(), connectors=[])
    assert not report.ok
    assert "no export connector" in report.summary()


def test_the_report_serializes_for_the_run_report():
    platforms = FakePlatforms(fail={"api.resend.com": 503})
    report = export_draft(
        TEXT,
        title=TITLE,
        approval=approved(),
        connectors=all_connectors(),
        transport=httpx.MockTransport(platforms),
    )
    data = report.to_dict()
    assert data["ok"] is False
    assert [r["connector"] for r in data["results"]] == ["buttondown", "beehiiv", "resend"]
    assert data["results"][2]["ok"] is False
    assert data["dropped"] == []


# --- keys never appear in a log ---------------------------------------------------------


@pytest.mark.parametrize(
    "fail",
    [
        {},
        {host: 401 for host in ("api.buttondown.com", "api.beehiiv.com", "api.resend.com")},
        {
            "api.buttondown.com": httpx.ConnectError(f"refused {KEYS['buttondown']}"),
            "api.beehiiv.com": httpx.ReadTimeout(f"slow {KEYS['beehiiv']}"),
            "api.resend.com": {"message": f"odd reply {KEYS['resend']}"},
        },
    ],
)
def test_keys_never_appear_in_a_log_or_the_report(fail, caplog):
    caplog.set_level(logging.DEBUG)
    platforms = FakePlatforms(fail=fail)
    report = export_draft(
        TEXT,
        title=TITLE,
        approval=approved(),
        connectors=all_connectors(),
        transport=httpx.MockTransport(platforms),
    )
    seen = caplog.text + json.dumps(report.to_dict()) + report.summary()
    seen += "".join(repr(c) for c in all_connectors())
    assert caplog.records, "export logs each connector's outcome"
    for key in KEYS.values():
        assert key not in seen


# --- settings from the environment ------------------------------------------------------


def test_each_connector_is_optional():
    assert connectors_from_env({}) == []
    only = connectors_from_env({"BUTTONDOWN_API_KEY": "k"})
    assert [c.name for c in only] == ["buttondown"]
    assert isinstance(only[0], Buttondown)


def test_connectors_read_their_keys_and_timeout_from_the_environment():
    beehiiv, resend = connectors_from_env(
        {k: v for k, v in ENV.items() if not k.startswith("BUTTONDOWN")}
        | {"NEWSLETTER_EXPORT_TIMEOUT": "12.5"}
    )
    assert isinstance(beehiiv, Beehiiv) and isinstance(resend, Resend)
    assert beehiiv.api_key == KEYS["beehiiv"] and beehiiv.publication_id == "pub_123"
    assert resend.segment_id == "seg_456"
    assert beehiiv.timeout == resend.timeout == 12.5
    assert connectors_from_env({"BUTTONDOWN_API_KEY": "k"})[0].timeout == 30.0


@pytest.mark.parametrize(
    "env, missing",
    [
        ({"BEEHIIV_API_KEY": "k"}, "BEEHIIV_PUBLICATION_ID"),
        ({"RESEND_API_KEY": "k", "RESEND_FROM": "a@example.org"}, "RESEND_SEGMENT_ID"),
        ({"RESEND_API_KEY": "k", "RESEND_SEGMENT_ID": "s"}, "RESEND_FROM"),
    ],
)
def test_a_connector_missing_a_required_setting_is_a_config_error(env, missing):
    with pytest.raises(ConfigError, match=missing):
        connectors_from_env(env)


def test_a_bad_timeout_is_a_config_error():
    with pytest.raises(ConfigError, match="NEWSLETTER_EXPORT_TIMEOUT"):
        connectors_from_env({"BUTTONDOWN_API_KEY": "k", "NEWSLETTER_EXPORT_TIMEOUT": "0"})
