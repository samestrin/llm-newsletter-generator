"""The email preview: `render_email(markdown)` and `GET /api/drafts/{id}/email`.

The preview is of the draft's current text, only after approval (409 before, with the
reason). The Markdown is untrusted — it can be edited by hand — so raw HTML comes out
escaped, a link is written only for a URL `safe_url` accepts, an image is shown as its
alt text, and neither the renderer nor the route makes any network call. Nothing here
sends mail; the app registers no route that sends.
"""

import html as html_escape
import re
import socket

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from newsletter.api import create_app
from newsletter.api.seed import seed_sample
from newsletter.api.store import DraftStore
from newsletter.render import render_email
from test_api import FakeRunner, headers, make_draft, make_settings


@pytest.fixture
def client(tmp_path):
    app = create_app(make_settings(tmp_path), runner=FakeRunner())
    with TestClient(app) as client:
        yield client


@pytest.fixture
def no_network(monkeypatch):
    """Any socket use or name lookup during the test is a failure, not a fetch."""

    def refuse(*args, **kwargs):
        raise AssertionError("the email preview made a network call")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def sample_text(tmp_path) -> str:
    assert seed_sample(tmp_path / "data")
    return DraftStore(tmp_path / "data").list()[0]["text"]


def approve(client, draft_id: str) -> None:
    response = client.post(f"/api/drafts/{draft_id}/approve", json={}, headers=headers())
    assert response.status_code == 200, response.text


# render_email


def test_the_sample_email_has_every_claim_link_and_flag_marker(tmp_path):
    text = sample_text(tmp_path)
    html = render_email(text)

    links = re.findall(r"\]\(<([^>]+)>\)", text)
    markers = re.findall(r"\*\*(Flagged:[^*]+)\*\*", text)
    numbers = re.findall(r"^- \\\[(\d+)\\\]", text, re.M)
    assert links and markers and numbers  # the sample carries all three
    for url in links:
        assert f'href="{html_escape.escape(url, quote=True)}"' in html, url
    for marker in markers:
        assert marker in html, marker
    for number in numbers:
        assert f"[{number}]" in html, number


def test_the_sample_email_keeps_the_run_report_table(tmp_path):
    text = sample_text(tmp_path)
    assert "| Step | Tier | Model |" in text
    html = render_email(text)
    assert "<table" in html
    assert "Tokens in" in html


def test_script_tags_and_javascript_links_come_out_escaped_and_unlinked():
    text = (
        "# Issue\n\n"
        '<script>alert("x")</script>\n\n'
        "<div onclick=\"steal()\">a block</div>\n\n"
        "[click](javascript:alert(1)) [upper](JaVaScRiPt:alert(1)) "
        "[tab](<java\tscript:alert(1)>) [data](data:text/html;base64,PHNjcmlwdD4=) "
        "[file](file:///etc/passwd) [ok](https://example.org/a).\n"
    )
    html = render_email(text)

    assert "<script" not in html
    assert "&lt;script&gt;" in html
    assert "<div" not in html
    assert "&lt;div onclick=" in html  # shown as escaped text, not an attribute
    assert 'href="javascript:' not in html.lower()
    assert 'href="java\tscript:' not in html
    assert 'href="data:' not in html
    assert 'href="file:' not in html
    assert 'href="https://example.org/a"' in html
    # The refused links are shown as their text, with no anchor: only the ok link is one.
    assert html.count("<a ") == 1
    for text_shown in ("click", "upper", "tab", "data", "file"):
        assert text_shown in html


def test_only_http_and_https_autolinks_are_linked():
    html = render_email("<https://example.org/a> <http://example.org/b> <ftp://example.org/c>\n")
    assert 'href="https://example.org/a"' in html
    assert 'href="http://example.org/b"' in html
    assert 'href="ftp:' not in html


def test_an_image_is_shown_as_its_alt_text_so_the_preview_fetches_nothing():
    html = render_email('![a tracker](https://tracker.example.org/pixel.gif "t")\n')
    assert "<img" not in html
    assert "a tracker" in html


def test_render_email_makes_no_network_call(no_network):
    text = (
        "# T\n\n[link](https://example.org/a) ![img](https://example.org/i.png)\n\n"
        "<https://example.org/b>\n"
    )
    assert 'href="https://example.org/a"' in render_email(text)


def test_the_preview_uses_the_issue_shell(tmp_path):
    html = render_email(sample_text(tmp_path))
    assert html.startswith("<!DOCTYPE html>")
    for marker in (
        '<html lang="en">',
        'name="viewport"',
        'class="container"',
        'width:600px;max-width:600px;',
        "Georgia,'Times New Roman',serif",
    ):
        assert marker in html, marker


def test_the_title_is_the_first_heading_escaped():
    html = render_email('# A <b>title</b> & more\n\ntext\n')
    assert "<title>A &lt;b&gt;title&lt;/b&gt; &amp; more</title>" in html
    assert "<title>Newsletter</title>" in render_email("no heading here\n")


def test_empty_text_still_renders_the_shell():
    assert render_email("").startswith("<!DOCTYPE html>")


# The route


def test_before_approval_the_route_is_409_with_the_reason(client):
    draft = make_draft(client)
    response = client.get(f"/api/drafts/{draft['id']}/email", headers=headers())
    assert response.status_code == 409
    assert "approve" in response.json()["detail"].lower()


def test_after_approval_the_route_returns_the_html_of_the_current_text(client):
    draft = make_draft(client)
    approve(client, draft["id"])
    response = client.get(f"/api/drafts/{draft['id']}/email", headers=headers())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"html"}
    assert body["html"] == render_email(draft["text"])


def test_an_edited_then_approved_drafts_email_shows_the_edit(client):
    draft = make_draft(client)
    approve(client, draft["id"])
    edited = draft["text"] + "\nAn editor added this line.\n"
    response = client.put(f"/api/drafts/{draft['id']}", json={"text": edited}, headers=headers())
    assert response.status_code == 200
    # The edit cleared the approval: 409 again until the new text is approved.
    assert client.get(f"/api/drafts/{draft['id']}/email", headers=headers()).status_code == 409
    approve(client, draft["id"])
    html = client.get(f"/api/drafts/{draft['id']}/email", headers=headers()).json()["html"]
    assert "An editor added this line." in html


def test_an_edited_draft_with_a_stale_approval_is_409(client):
    """The approval on file is of other text: the preview still refuses it."""
    draft = make_draft(client)
    approve(client, draft["id"])
    store = DraftStore(client.app.state.settings.data_dir)
    stored = store.get(draft["id"])
    stored["text"] += "\nchanged without clearing the approval\n"
    store.files.write(draft["id"], stored)
    assert client.get(f"/api/drafts/{draft['id']}/email", headers=headers()).status_code == 409


def test_the_email_route_needs_the_token(client):
    draft = make_draft(client)
    response = client.get(f"/api/drafts/{draft['id']}/email")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_an_unknown_draft_is_404(client):
    assert client.get(f"/api/drafts/{'0' * 32}/email", headers=headers()).status_code == 404


def test_the_email_route_makes_no_network_call(client, no_network, tmp_path):
    draft = make_draft(client)
    approve(client, draft["id"])
    response = client.get(f"/api/drafts/{draft['id']}/email", headers=headers())
    assert response.status_code == 200
    assert response.json()["html"].startswith("<!DOCTYPE html>")


def test_the_sample_drafts_email_serves_every_link_and_marker(client, tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, runner=FakeRunner(), seed_sample=True)
    with TestClient(app) as seeded:
        draft = seeded.get("/api/drafts", headers=headers()).json()[0]
        approve(seeded, draft["id"])
        html = seeded.get(f"/api/drafts/{draft['id']}/email", headers=headers()).json()["html"]
    text = sample_text(tmp_path / "other")
    for url in re.findall(r"\]\(<([^>]+)>\)", text):
        assert f'href="{html_escape.escape(url, quote=True)}"' in html, url
    for marker in re.findall(r"\*\*(Flagged:[^*]+)\*\*", text):
        assert marker in html, marker


def test_the_app_registers_no_route_that_sends(client):
    """The whole route table, exactly: reads, the run start, the edit and the approval.
    A new route — anything that could send — fails this test until it is reviewed."""
    registered = {
        (method, route.path)
        for route in client.app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
    }
    assert registered == {
        ("GET", "/api/docs"),
        ("GET", "/api/docs/scalar.js"),
        ("GET", "/api/health"),
        ("POST", "/api/runs"),
        ("GET", "/api/runs/{run_id}"),
        ("GET", "/api/drafts"),
        ("GET", "/api/drafts/{draft_id}"),
        ("PUT", "/api/drafts/{draft_id}"),
        ("POST", "/api/drafts/{draft_id}/approve"),
        ("GET", "/api/drafts/{draft_id}/email"),
    }
    assert not any("send" in path.lower() for _, path in registered)
