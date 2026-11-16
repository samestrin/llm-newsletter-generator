"""The API reference at `/api/docs`: Scalar, vendored, offline, reading `/api/openapi.json`.

The spec declares every route with a response schema and the bearer-token guard as a
security scheme. The spec and the reference are readable without the token; only the
routes are guarded.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from test_api import CONTENT, FakeRunner, headers, make_draft, make_settings

from newsletter.api import create_app

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "newsletter" / "api" / "static"
SCALAR = STATIC / "scalar" / "standalone.js"
SCRIPT = "/api/docs/scalar.js"
SPEC = "/api/openapi.json"

GUARDED = {
    ("post", "/api/runs"),
    ("get", "/api/runs/{run_id}"),
    ("get", "/api/drafts"),
    ("get", "/api/drafts/{draft_id}"),
    ("put", "/api/drafts/{draft_id}"),
    ("post", "/api/drafts/{draft_id}/approve"),
}
WRITES = {
    ("post", "/api/runs"),
    ("put", "/api/drafts/{draft_id}"),
    ("post", "/api/drafts/{draft_id}/approve"),
}


@pytest.fixture
def client(tmp_path):
    app = create_app(make_settings(tmp_path), runner=FakeRunner())
    with TestClient(app) as client:
        yield client


@pytest.fixture
def spec(client):
    response = client.get(SPEC)
    assert response.status_code == 200
    return response.json()


def _assets(html: str) -> list[str]:
    return re.findall(r'(?:src|href)="([^"]+)"', html)


def _configuration(html: str) -> dict:
    (attribute,) = re.findall(r"data-configuration='([^']*)'", html)
    return json.loads(attribute)


def _operations(spec: dict) -> dict[tuple[str, str], dict]:
    return {
        (method, path): operation
        for path, item in spec["paths"].items()
        for method, operation in item.items()
        if method in {"get", "post", "put", "delete", "patch"}
    }


def _success(operation: dict) -> dict:
    code = "202" if "202" in operation["responses"] else "200"
    return operation["responses"][code]


def test_docs_is_html_that_points_at_the_spec_and_the_vendored_script(client):
    response = client.get("/api/docs")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert SPEC in response.text
    assert SCRIPT in _assets(response.text)


def test_the_reference_loads_nothing_from_another_origin(client):
    html = client.get("/api/docs").text
    assets = _assets(html)

    assert assets
    for asset in assets:
        assert asset.startswith("/"), asset  # no CDN, no protocol-relative URL
        assert client.get(asset).status_code == 200, asset
    assert not re.search(r"https?:", html, re.IGNORECASE)
    assert not re.search(r"""(?:src|href|action)\s*=\s*["']?//""", html)


def test_the_configuration_reads_this_spec_and_turns_off_every_scalar_service(client):
    config = _configuration(client.get("/api/docs").text)

    assert config["url"] == SPEC
    assert config["proxyUrl"] is None  # "Test Request" calls this API, not proxy.scalar.com
    assert config["withDefaultFonts"] is False  # no font is fetched
    assert config["telemetry"] is False
    assert config["hideClientButton"] is True  # no button opens client.scalar.com


def test_the_vendored_script_is_served_by_this_api(client):
    response = client.get(SCRIPT)

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.content == SCALAR.read_bytes()


def test_the_vendored_script_is_the_pinned_release():
    """The file is the published bundle of the version named beside it, byte for byte."""
    notes = (STATIC / "scalar" / "README.md").read_text()
    versions = set(re.findall(r"@scalar/api-reference@(\d+\.\d+\.\d+)", notes))
    (digest,) = re.findall(r"\b([0-9a-f]{64})\b", notes)

    assert versions == {"1.78.0"}  # the same pinned version as llm-pdf-ocr-api
    assert hashlib.sha256(SCALAR.read_bytes()).hexdigest() == digest


def test_the_theme_is_the_apps_accent_in_light_and_dark(client):
    """The reference's accent is web/src/styles.css's --color-accent, in both modes."""
    styles = (ROOT / "web" / "src" / "styles.css").read_text()
    light = re.search(r":root\s*{[^}]*--color-accent:\s*(#[0-9a-f]{6})", styles).group(1)
    dark_block = re.search(r"@media \(prefers-color-scheme: dark\)\s*{(.*)}", styles, re.S).group(1)
    dark = re.search(r"--color-accent:\s*(#[0-9a-f]{6})", dark_block).group(1)
    on_light = re.search(r":root\s*{[^}]*--color-on-accent:\s*(#[0-9a-f]{6})", styles).group(1)
    on_dark = re.search(r"--color-on-accent:\s*(#[0-9a-f]{6})", dark_block).group(1)

    html = client.get("/api/docs").text
    light_block = re.search(r"\.light-mode\s*{([^}]*)}", html).group(1)
    dark_block = re.search(r"\.dark-mode\s*{([^}]*)}", html).group(1)
    for block, accent, on_accent in ((light_block, light, on_light), (dark_block, dark, on_dark)):
        assert f"--scalar-color-accent: {accent};" in block
        assert f"--scalar-button-1: {accent};" in block
        assert f"--scalar-button-1-color: {on_accent};" in block


def test_the_docs_and_the_script_are_not_in_the_schema(spec):
    assert "/api/docs" not in spec["paths"]
    assert SCRIPT not in spec["paths"]


def test_the_old_docs_pages_are_gone(client):
    for path in ("/docs", "/redoc", "/api/redoc"):
        assert client.get(path).status_code == 404, path


def test_the_spec_declares_no_servers_prefix(spec):
    # nginx passes /api/ through unchanged, so "Test Request" reaches the routes as is.
    assert not spec.get("servers")


def test_every_registered_route_is_in_the_spec_with_a_success_schema(client, spec):
    operations = _operations(spec)
    registered = {
        (method.lower(), route.path)
        for route in client.app.routes
        if isinstance(route, APIRoute) and route.include_in_schema
        for method in route.methods
    }

    assert registered == set(operations)
    assert registered == GUARDED | {("get", "/api/health")}
    for operation in operations.values():
        schema = _success(operation)["content"]["application/json"]["schema"]
        assert schema, operation  # a $ref or an inline schema, never empty


def test_the_guarded_routes_show_the_bearer_security_scheme(spec):
    schemes = spec["components"]["securitySchemes"]
    (name, scheme) = next(iter(schemes.items()))
    assert scheme == {"type": "http", "scheme": "bearer"}

    for key, operation in _operations(spec).items():
        if key in GUARDED:
            assert operation["security"] == [{name: []}], key
        else:
            assert not operation.get("security"), key


def test_the_error_responses_are_declared(spec):
    for key, operation in _operations(spec).items():
        responses = operation["responses"]
        if key in GUARDED:
            assert "401" in responses, key
        if key in WRITES:
            for code in ("400", "403", "415", "422", "503"):
                assert code in responses, (key, code)
        if key in {("post", "/api/runs"), ("post", "/api/drafts/{draft_id}/approve")}:
            assert "409" in responses, key
        if "{run_id}" in key[1] or "{draft_id}" in key[1]:
            assert "404" in responses, key
        for code, response in responses.items():
            if code.startswith(("4", "5")):
                assert response["content"]["application/json"]["schema"], (key, code)


def test_the_spec_and_the_reference_need_no_token(client):
    assert client.get(SPEC).status_code == 200
    assert client.get("/api/docs").status_code == 200
    assert client.get(SCRIPT).status_code == 200


def test_the_response_models_do_not_change_the_bodies(client):
    """A draft's flags and report pass through exactly as stored, partial ones included."""
    draft = make_draft(client)

    assert draft["flags"] == CONTENT["flags"]
    assert draft["report"] == CONTENT["report"]
    listed = client.get("/api/drafts", headers=headers()).json()
    assert listed[0]["flags"] == len(CONTENT["flags"])
    assert "text" not in listed[0]


def test_the_spec_names_the_service_and_its_version(spec):
    from newsletter import __version__

    assert spec["info"]["title"] == "newsletter"
    assert spec["info"]["version"] == __version__
