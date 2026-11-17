"""Spec completeness: every route the app registers (except `include_in_schema=False`)
is in the OpenAPI spec with a response schema — proven both ways with fixtures.

The proxy half of the contract (through the web container, `/api/openapi.json` answers
JSON and `/api/docs` the Scalar page, not the SPA's `index.html`) is checked by the
`compose` job of `.github/workflows/ci.yml`, where nginx's SPA fallback would otherwise
hide a miss behind a 200 `index.html`.
"""

import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from newsletter.api import create_app
from spec_completeness import spec_gaps
from test_api import FakeRunner, make_settings


@pytest.fixture
def app(tmp_path):
    return create_app(make_settings(tmp_path), runner=FakeRunner())


def test_the_apps_spec_is_complete(app):
    assert spec_gaps(app) == []


def test_a_route_added_without_a_response_schema_is_caught(app):
    """The done-when fixture: a new route with no response model and no return
    annotation gets a 200 with no content in the spec, and the check names it."""

    @app.get("/api/extra")
    def extra():
        return {"ok": True}

    gaps = spec_gaps(app)
    assert len(gaps) == 1
    assert "GET /api/extra" in gaps[0]
    assert "schema" in gaps[0]


def test_a_route_added_with_a_response_schema_passes(app):
    class Extra(BaseModel):
        ok: bool

    @app.get("/api/extra", response_model=Extra)
    def extra() -> Extra:
        return Extra(ok=True)

    assert spec_gaps(app) == []


def test_a_route_hidden_from_the_schema_is_ignored(app):
    @app.get("/api/hidden", include_in_schema=False)
    def hidden():
        return {"ok": True}

    assert spec_gaps(app) == []


def test_a_route_missing_from_the_spec_or_without_a_success_response_is_caught():
    """The other two ways a spec can be incomplete, on a crafted spec: the path is
    absent, or the operation declares no 2xx response at all."""
    app = FastAPI()

    @app.get("/api/absent")
    def absent():
        return {"ok": True}

    @app.get("/api/no-success")
    def no_success():
        return {"ok": True}

    app.openapi = lambda: {
        "paths": {
            "/api/no-success": {"get": {"responses": {"401": {"description": "Missing token."}}}}
        }
    }

    gaps = spec_gaps(app)
    assert len(gaps) == 2
    assert "GET /api/absent" in gaps[0] and "not in the spec" in gaps[0]
    assert "GET /api/no-success" in gaps[1] and "success response" in gaps[1]
