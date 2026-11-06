"""The HTTP API: start a run, read its status, list, read, edit and approve drafts.

Every request but `GET /api/health` needs `Authorization: Bearer $NEWSLETTER_API_TOKEN`.
A write (any method but GET, HEAD and OPTIONS) also needs an `Origin` in the allowed
list and a JSON object body sent as `application/json`, so a page on another site
cannot make one even with the token. Checks run in that order: 401, 403, then 415 or
400. No error body echoes the request, and nothing returned or stored holds a secret.
"""

import datetime as dt
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


def create_app(
    settings: ApiSettings,
    *,
    runner: Runner | None = None,
    config: SourcesConfig | None = None,
) -> FastAPI:
    """The API. A run that names no feeds reads the sources in `config` (`serve --config`)."""
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
        yield

    app = FastAPI(title="newsletter", docs_url=None, redoc_url=None, openapi_url=None)
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

    async def guard(request: Request) -> dict[str, Any] | None:
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

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/runs", status_code=202)
    def start_run(raw: Guarded) -> dict[str, Any]:
        body = parse(RunBody, raw)
        request = _run_request(body, config)
        try:
            return jobs.start(request)
        except Busy as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/runs/{run_id}", dependencies=[Depends(guard)])
    def get_run(run_id: str) -> dict[str, Any]:
        return _found(runs.get, run_id)

    @app.get("/api/drafts", dependencies=[Depends(guard)])
    def list_drafts() -> list[dict[str, Any]]:
        return [_summary(draft) for draft in drafts.list()]

    @app.get("/api/drafts/{draft_id}", dependencies=[Depends(guard)])
    def get_draft(draft_id: str) -> dict[str, Any]:
        return _view(_found(drafts.get, draft_id))

    @app.put("/api/drafts/{draft_id}")
    def edit_draft(draft_id: str, raw: Guarded) -> dict[str, Any]:
        body = parse(EditBody, raw)
        return _view(_found(drafts.edit, draft_id, body.text))

    @app.post("/api/drafts/{draft_id}/approve")
    def approve_draft(draft_id: str, raw: Guarded) -> dict[str, Any]:
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
