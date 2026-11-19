"""The HTTP API: start a run, read its status, list, read, edit and approve drafts.

Every request but `GET /api/health` needs `Authorization: Bearer $NEWSLETTER_API_TOKEN`.
A write (any method but GET, HEAD and OPTIONS) also needs an `Origin` in the allowed
list and a JSON object body sent as `application/json`, so a page on another site
cannot make one even with the token. Checks run in that order: 401, 403, then 415 or
400. No error body echoes the request, and nothing returned or stored holds a secret.

`GET /api/openapi.json` is the OpenAPI spec and `GET /api/docs` the Scalar reference
that reads it (vendored in `newsletter/api/static/`, no CDN). Both are readable without
the token: the spec holds route names and schemas only, and the reference's "Test
Request" still needs the token, declared there as the bearer security scheme. FastAPI's
own docs pages and ReDoc stay off.
"""

import datetime as dt
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from newsletter import __version__
from newsletter.api import seed
from newsletter.api.jobs import Busy, Jobs, Runner
from newsletter.api.pipeline import RunRequest, default_runner
from newsletter.api.settings import ApiSettings
from newsletter.api.store import (
    Conflict,
    DraftStore,
    NotFound,
    RunStore,
    is_approved,
    redactor,
)
from newsletter.config import SourcesConfig

log = logging.getLogger(__name__)

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
Body = TypeVar("Body", bound=BaseModel)
MAX_FEEDS = 50


class RunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feeds: list[str] = Field(default_factory=list, max_length=MAX_FEEDS)
    """Empty to read the sources config `serve` was started with."""
    title: str = Field("Newsletter", min_length=1, max_length=200)
    topic: str | None = Field(None, min_length=1, max_length=200)
    """Default: the sources config's topic, else "news"."""
    date: dt.date | None = None


class EditBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=1_000_000)


class ApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    """The sha256 of the text the reviewer read; the approval is refused if it changed."""


# Response models. `extra="allow"` and `response_model_exclude_unset=True` on every
# route keep the bodies exactly as they were before the spec existed: a stored draft
# or report passes through whole, including fields these models do not name.


class Health(BaseModel):
    ok: bool


class RunView(BaseModel):
    """A run's status: `queued`, `running`, then `done` with its draft id or `error`."""

    model_config = ConfigDict(extra="allow")

    id: str
    status: Literal["queued", "running", "done", "error"]
    error: str | None
    draft_id: str | None
    feeds: list[str]
    title: str
    created_at: str
    started_at: str | None
    finished_at: str | None


class FlaggedClaim(BaseModel):
    """A claim shown in the issue, marked: its source does not support it, or it was
    not checked."""

    model_config = ConfigDict(extra="allow")

    id: str
    story_id: str | None = None
    text: str | None = None
    status: str | None = None
    reason: str | None = None
    source_url: str | None = None


class RunReport(BaseModel):
    """Everything that failed, was left out or is in doubt, and what it cost."""

    model_config = ConfigDict(extra="allow")

    ok: bool | None = None
    failed_sources: list[dict[str, Any]] | None = None
    linked_articles: list[dict[str, Any]] | None = None
    left_out: list[dict[str, Any]] | None = None
    grouping_error: str | None = None
    dropped_stories: list[dict[str, Any]] | None = None
    dropped_claims: list[dict[str, Any]] | None = None
    flagged_claims: list[dict[str, Any]] | None = None
    editor_error: str | None = None
    editor_rejections: list[str] | None = None
    steps: list[dict[str, Any]] | None = None
    total_cost: float | None = None
    cost_complete: bool | None = None
    failed_steps: list[str] | None = None
    voice: dict[str, Any] | None = None


class Approval(BaseModel):
    sha256: str
    approved_at: str


class DraftSummary(BaseModel):
    """A draft as listed: no text, the flag count instead of the flags."""

    model_config = ConfigDict(extra="allow")

    id: str
    run_id: str
    title: str
    issue_date: str
    created_at: str
    updated_at: str
    flags: int
    approved: bool


class DraftView(BaseModel):
    """A draft whole: its text, flags, run report and approval, if any."""

    model_config = ConfigDict(extra="allow")

    id: str
    run_id: str
    title: str
    issue_date: str
    text: str
    flags: list[FlaggedClaim] = Field(default_factory=list)
    report: RunReport = Field(default_factory=RunReport)
    approval: Approval | None = None
    created_at: str
    updated_at: str
    approved: bool


class ErrorBody(BaseModel):
    """An error: the reason only, never the request."""

    detail: str


class ValidationProblem(BaseModel):
    loc: list[str]
    msg: str


class InvalidRequest(BaseModel):
    """422: a field is missing or invalid (a list of problems), or the run names no
    feed and `serve` was started without a config (a plain reason)."""

    detail: str | list[ValidationProblem]


UNAUTHORIZED = {401: {"description": "Missing or wrong bearer token.", "model": ErrorBody}}
NOT_FOUND = {404: {"description": "No draft or run with this id.", "model": ErrorBody}}
WRITE_ERRORS = {
    **UNAUTHORIZED,
    400: {"description": "The body is not JSON, or not a JSON object.", "model": ErrorBody},
    403: {"description": "This origin may not write.", "model": ErrorBody},
    415: {"description": "The body was not sent as application/json.", "model": ErrorBody},
    422: {"description": "A field is missing or invalid.", "model": InvalidRequest},
    503: {
        "description": "Writes are off: NEWSLETTER_API_TOKEN is not set.",
        "model": ErrorBody,
    },
}

STATIC = Path(__file__).parent / "static"


def create_app(
    settings: ApiSettings,
    *,
    runner: Runner | None = None,
    config: SourcesConfig | None = None,
    seed_sample: bool = False,
) -> FastAPI:
    """The API. A run that names no feeds reads the sources in `config` (`serve --config`).
    With `seed_sample`, the committed sample draft is written at start, once (see
    `newsletter.api.seed`)."""
    redact = redactor(settings.secrets)
    drafts = DraftStore(settings.data_dir)
    runs = RunStore(settings.data_dir)
    jobs = Jobs(
        runner or default_runner(settings.env),
        runs,
        drafts,
        timeout=settings.run_timeout,
        redact=redact,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runs.mark_interrupted()
        if seed_sample:
            seed.seed_sample(settings.data_dir)
        yield

    app = FastAPI(
        title="newsletter",
        version=__version__,
        description=(
            "Generate text-only newsletters from RSS feeds using LLMs: start a run, read "
            "its status, list, read, edit and approve drafts. Every route but "
            "`/api/health` needs the bearer token. The spec and the reference at "
            "`/api/docs` are readable without it; they hold route names and schemas only."
        ),
        docs_url=None,  # /api/docs is the Scalar reference below; /redoc stays off
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.router.lifespan_context = lifespan
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.config = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.origins),
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Declares the bearer scheme in the spec so the reference can send the token.
    # auto_error=False: the guard below keeps answering 401 itself, in its own order.
    bearer = HTTPBearer(
        auto_error=False,
        description="The `NEWSLETTER_API_TOKEN` the server was started with.",
    )

    async def guard(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> dict[str, Any] | None:
        if request.method not in SAFE_METHODS and not settings.token:
            raise HTTPException(503, "writes are off: set NEWSLETTER_API_TOKEN")
        if not _token_ok(request.headers.get("authorization"), settings.token):
            raise HTTPException(
                401, "missing or wrong token", headers={"WWW-Authenticate": "Bearer"}
            )
        if request.method in SAFE_METHODS:
            return None
        if request.headers.get("origin") not in settings.origins:
            raise HTTPException(403, "this origin may not write")
        content_type = request.headers.get("content-type", "")
        if content_type.split(";")[0].strip().lower() != "application/json":
            raise HTTPException(415, "send the body as application/json")
        try:
            body = json.loads(await request.body())
        except ValueError:
            raise HTTPException(400, "the body is not JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "the body must be a JSON object")
        return body

    Guarded = Annotated[dict[str, Any], Depends(guard)]

    def parse(model: type[Body], body: dict[str, Any]) -> Body:
        # Read here, after the guard: a body parameter would be read before it.
        try:
            return model.model_validate(body)
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from None

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        # Field and reason only: the default handler echoes the input back.
        problems = [
            {"loc": [str(part) for part in e["loc"]], "msg": e["msg"]} for e in error.errors()
        ]
        return JSONResponse({"detail": redact(problems)}, status_code=422)

    @app.get("/api/docs", include_in_schema=False)
    def api_reference() -> HTMLResponse:
        """The API reference: Scalar, vendored in `static/`, reading `/api/openapi.json`."""
        return HTMLResponse((STATIC / "docs.html").read_text(encoding="utf-8"))

    @app.get("/api/docs/scalar.js", include_in_schema=False)
    def scalar_script() -> FileResponse:
        """The vendored Scalar bundle, served by this API so the page loads no CDN."""
        return FileResponse(STATIC / "scalar" / "standalone.js", media_type="text/javascript")

    @app.get("/api/health", response_model=Health)
    def health() -> dict[str, bool]:
        """Report that the service is up. No token needed."""
        return {"ok": True}

    @app.post(
        "/api/runs",
        status_code=202,
        response_model=RunView,
        response_model_exclude_unset=True,
        responses={
            202: {"description": "The run is queued; follow it at `/api/runs/{id}`."},
            **WRITE_ERRORS,
            409: {"description": "A run is already going.", "model": ErrorBody},
        },
    )
    def start_run(raw: Guarded) -> dict[str, Any]:
        body = parse(RunBody, raw)
        request = _run_request(body, config)
        try:
            return jobs.start(request)
        except Busy as error:
            raise HTTPException(409, str(error)) from None

    @app.get(
        "/api/runs/{run_id}",
        dependencies=[Depends(guard)],
        response_model=RunView,
        response_model_exclude_unset=True,
        responses={**UNAUTHORIZED, **NOT_FOUND},
    )
    def get_run(run_id: str) -> dict[str, Any]:
        """A run's status, and its draft id once it is done."""
        return _found(runs.get, run_id)

    @app.get(
        "/api/drafts",
        dependencies=[Depends(guard)],
        response_model=list[DraftSummary],
        response_model_exclude_unset=True,
        responses=UNAUTHORIZED,
    )
    def list_drafts() -> list[dict[str, Any]]:
        """Every draft, newest first, without its text."""
        return [_summary(draft) for draft in drafts.list()]

    @app.get(
        "/api/drafts/{draft_id}",
        dependencies=[Depends(guard)],
        response_model=DraftView,
        response_model_exclude_unset=True,
        responses={**UNAUTHORIZED, **NOT_FOUND},
    )
    def get_draft(draft_id: str) -> dict[str, Any]:
        """A draft whole: its text, flags, run report and approval."""
        return _view(_found(drafts.get, draft_id))

    @app.put(
        "/api/drafts/{draft_id}",
        response_model=DraftView,
        response_model_exclude_unset=True,
        responses={**WRITE_ERRORS, **NOT_FOUND},
    )
    def edit_draft(draft_id: str, raw: Guarded) -> dict[str, Any]:
        """Replace the draft's text. Any edit clears the approval."""
        body = parse(EditBody, raw)
        return _view(_found(drafts.edit, draft_id, body.text))

    @app.post(
        "/api/drafts/{draft_id}/approve",
        response_model=DraftView,
        response_model_exclude_unset=True,
        responses={
            **WRITE_ERRORS,
            **NOT_FOUND,
            409: {
                "description": "The draft changed since it was read.",
                "model": ErrorBody,
            },
        },
    )
    def approve_draft(draft_id: str, raw: Guarded) -> dict[str, Any]:
        """Approve the draft's text as it is now; with `sha256`, only if it is that text."""
        body = parse(ApproveBody, raw)
        try:
            return _view(_found(drafts.approve, draft_id, body.sha256))
        except Conflict as error:
            raise HTTPException(409, str(error)) from None

    return app


def _token_ok(header: str | None, token: str | None) -> bool:
    if not token or not header:
        return False
    scheme, _, sent = header.partition(" ")
    if scheme.lower() != "bearer" or not sent:
        return False
    return hmac.compare_digest(sent.encode("utf-8"), token.encode("utf-8"))


def _found(lookup, item_id: str, *args):
    try:
        return lookup(item_id, *args)
    except NotFound:
        raise HTTPException(404, "not found") from None


def _view(draft: dict[str, Any]) -> dict[str, Any]:
    return {**draft, "approved": is_approved(draft)}


def _summary(draft: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "run_id", "title", "issue_date", "created_at", "updated_at")
    return {
        **{key: draft.get(key) for key in keys},
        "flags": len(draft.get("flags", [])),
        "approved": is_approved(draft),
    }


def _run_request(body: RunBody, config: SourcesConfig | None) -> RunRequest:
    """The run `body` asks for. Feeds in the body replace every source in `config`."""
    topic = body.topic or (config.topic if config else "news")
    per_feed = config.per_feed if config else None
    if body.feeds:
        return RunRequest(body.feeds, body.title, topic, body.date, per_feed=per_feed)
    if config is None:
        raise HTTPException(422, "give at least one feed, or start serve with --config")
    return RunRequest(
        list(config.feeds),
        body.title,
        topic,
        body.date,
        pages=list(config.pages),
        hackernews=config.hackernews,
        per_feed=per_feed,
    )
