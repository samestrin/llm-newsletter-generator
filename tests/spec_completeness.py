"""The spec-completeness check: every route the app registers (except
`include_in_schema=False`) must be in the OpenAPI spec with a response schema.

`tests/test_spec_completeness.py` proves it both ways with fixtures. The proxy half of
the contract — through the web container, `/api/openapi.json` answers JSON and
`/api/docs` the Scalar page, not the SPA's `index.html` — is checked by the `compose`
job of `.github/workflows/ci.yml`.
"""

from fastapi import FastAPI
from fastapi.routing import APIRoute


def spec_gaps(app: FastAPI) -> list[str]:
    """One line per registered route the spec does not cover: absent from `paths`,
    without a success (2xx) response, or without a JSON schema on that response.
    Empty when the spec is complete. Routes with `include_in_schema=False` are the
    app's own choice to hide (the reference page and its script) and are skipped."""
    spec = app.openapi()
    gaps = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        item = spec.get("paths", {}).get(route.path, {})
        for method in sorted(route.methods):
            key = f"{method} {route.path}"
            operation = item.get(method.lower())
            if operation is None:
                gaps.append(f"{key}: registered but not in the spec")
                continue
            success = [
                response
                for code, response in operation.get("responses", {}).items()
                if code.startswith("2")
            ]
            if not success:
                gaps.append(f"{key}: no success response in the spec")
                continue
            if not any(
                response.get("content", {}).get("application/json", {}).get("schema")
                for response in success
            ):
                gaps.append(f"{key}: success response has no response schema")
    return gaps
