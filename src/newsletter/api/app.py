"""The HTTP API: start a run, read its status, list, read, edit and approve drafts,
and preview an approved draft as an HTML email. Nothing here sends mail.

Every request but `GET /api/health` needs `Authorization: Bearer $NEWSLETTER_API_TOKEN`.
A write (any method but GET, HEAD and OPTIONS) also needs an `Origin` in the allowed
list and a JSON object body sent as `application/json`, so a page on another site
cannot make one even with the token. Checks run in that order: 401, 403, then 415 or
400. No error body echoes the request, but for the feeds of a run none of which could be
read, each named with why; nothing returned or stored holds a secret.

`POST /api/feeds/check` reads one address as a run would and says whether it is a feed a
run can read, capped at `CHECKS_PER_MINUTE` calls a minute; `POST /api/runs` checks every
feed it names the same way first. With `NEWSLETTER_DEMO=1` a run takes at most
`DEMO_MAX_FEEDS` feeds and `DEMO_PER_FEED` items per feed, and its `date` is ignored.

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
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from newsletter import __version__
from newsletter.api import seed
from newsletter.api.credentials import PROVIDERS, Credentials, public_providers, request_credentials
from newsletter.api.feed_check import RateLimit, check_feed
from newsletter.api.jobs import BUSY, Busy, Jobs, Runner
from newsletter.api.pipeline import RunRequest, default_runner, run_settings
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
from newsletter.llm import ConfigError
from newsletter.render import render_email
from newsletter.sources.http import Fetch, Resolve, http_get, resolve_host
from newsletter.sources.rss import FeedFailure

log = logging.getLogger(__name__)

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
Body = TypeVar("Body", bound=BaseModel)
MAX_FEEDS = 50
DEMO_MAX_FEEDS = 6
DEMO_PER_FEED = 3
MAX_FOCUS = 300
CHECKS_PER_MINUTE = 30
"""`POST /api/feeds/check` calls a minute, every caller together (owner, 2026-11-20)."""
BUSY_RETRY_AFTER = 30
"""Seconds a caller is told to wait when a run is already going."""
PRESETS = json.loads((Path(__file__).parent / "presets.json").read_text(encoding="utf-8"))


class RunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feeds: list[str] = Field(default_factory=list, max_length=MAX_FEEDS)
    """Empty to read the sources config `serve` was started with."""
    title: str = Field("Newsletter", min_length=1, max_length=200)
    topic: str | None = Field(None, min_length=1, max_length=200)
    """Default: the sources config's topic, else "news"."""
    focus: str | None = Field(None, max_length=MAX_FOCUS)
    """What the issue should put the emphasis on, e.g. "practical AI tools for small
    businesses": ranked by in place of the topic and stated to the editor, which may lead
    with and favour matching stories but adds no story or claim for it. Read as one line;
    blank is none."""
    date: dt.date | None = None
    """Ignored in demo mode."""
    per_feed: int | None = Field(None, ge=1, le=100)
    """Items to read from each feed; default the sources config's, else every item. In
    demo mode at most 3."""

    @field_validator("focus")
    @classmethod
    def _one_line(cls, focus: str | None) -> str | None:
        if focus is None:
            return None
        return " ".join(focus.split()) or None


class CheckBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2000)


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


class ProgressStep(BaseModel):
    """One step of a run: `waiting`, `running`, then `done`, `failed` or `skipped`.

    `detail` says what the step did, in words built from counts, never a model's reply.
    """

    id: str
    label: str
    state: Literal["waiting", "running", "done", "failed", "skipped"]
    done: int | None
    total: int | None
    detail: str | None
    started_at: str | None
    ended_at: str | None


class RunView(BaseModel):
    """A run's status: `queued`, `running`, then `done` with its draft id or `error`,
    and the progress of each of its steps."""

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
    progress: list[ProgressStep] = Field(default_factory=list)
    """Empty for a run from before progress was kept."""
    editor_message: str | None = None
    """When the editor failed in a finished run, one plain sentence saying so; the raw
    error is only in the draft's report, as `editor_error`."""


class FlaggedClaim(BaseModel):
    """A claim shown in the issue, marked: its source does not support it, or it was
    not checked."""

    model_config = ConfigDict(extra="allow")

    id: str
    story_id: str | None = None
    text: str | None = None
    status: str | None = None
    reason: str | None = None
    """The check's own reason: the model's, or why the check failed."""
    source_url: str | None = None
    passage: str | None = None
    """The source text the claim was tied to. Absent from drafts made before it was kept."""
    explanation: str | None = None
    """Why it is flagged, in plain words built in code. Absent from older drafts."""


class RunReport(BaseModel):
    """Everything that failed, was left out or is in doubt, and what it cost."""

    model_config = ConfigDict(extra="allow")

    ok: bool | None = None
    focus: str | None = None
    """The emphasis the run was asked for; absent when it had none."""
    failed_sources: list[dict[str, Any]] | None = None
    linked_articles: list[dict[str, Any]] | None = None
    left_out: list[dict[str, Any]] | None = None
    grouping_error: str | None = None
    grouping_note: str | None = None
    """Why stories were not grouped by meaning: the provider has no embedding model."""
    dropped_stories: list[dict[str, Any]] | None = None
    dropped_claims: list[dict[str, Any]] | None = None
    flagged_claims: list[dict[str, Any]] | None = None
    editor_error: str | None = None
    editor_message: str | None = None
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
    focus: str | None = None
    """The emphasis the run was asked for; absent when it had none, and on older drafts."""


class ProviderView(BaseModel):
    """A model provider a visitor may pick: send its `id` as `X-Model-Preset` with a key
    as `X-Model-Key`, or `demo` with `X-Demo-Code`. No secret: the file holds none."""

    id: str
    label: str
    kind: Literal["key", "demo"]
    base_url: str | None
    """`None` for the demo provider: a demo code goes to the demo gateway."""
    fast_model: str
    reasoning_model: str
    embedding_model: str | None
    """`None` when the provider has no embedding model."""


class PresetView(BaseModel):
    """A set of public feeds the run form can fill in."""

    id: str
    label: str
    feeds: list[str]


class FeedCheckView(BaseModel):
    """Whether a run can read the address: it answered, parses as RSS or Atom and has at
    least one item. When not, `error` says why in one plain sentence."""

    ok: bool
    title: str | None
    """The feed's own title, when `ok`."""
    items: int
    error: str | None


class ErrorBody(BaseModel):
    """An error: the reason only, never the request."""

    detail: str


class RetryLater(BaseModel):
    """Not now: ask again in `retry_after` seconds, also sent as `Retry-After`."""

    detail: str
    retry_after: int


class FeedProblem(BaseModel):
    url: str
    error: str
    """The check's plain sentence."""


class EmailPreview(BaseModel):
    """The approved draft's current text as an HTML email preview. Nothing is sent."""

    html: str


class ValidationProblem(BaseModel):
    loc: list[str]
    msg: str


class InvalidRequest(BaseModel):
    """422: a field is missing or invalid (a list of problems), the run names no feed and
    `serve` was started without a config, or too many feeds for demo mode (a plain
    reason), or no feed the run names could be read (a plain reason and `errors`)."""

    detail: str | list[ValidationProblem]
    errors: list[FeedProblem] | None = None
    """When no feed could be read: each feed and why."""


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

RETRY_AFTER_HEADER = {
    "Retry-After": {
        "description": "Seconds to wait before asking again, as `retry_after`.",
        "schema": {"type": "integer"},
    }
}
RUN_GOING = {
    409: {
        "description": "A run is already going.",
        "model": RetryLater,
        "headers": RETRY_AFTER_HEADER,
    }
}
TOO_MANY = {
    429: {
        "description": f"Over {CHECKS_PER_MINUTE} checks a minute.",
        "model": RetryLater,
        "headers": RETRY_AFTER_HEADER,
    }
}

STATIC = Path(__file__).parent / "static"


class Answer(Exception):
    """An error answer with more than `detail`: sent as `body` with `headers`."""

    def __init__(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None):
        self.status = status
        self.body = body
        self.headers = headers or {}


def retry_later(status: int, detail: str, seconds: int) -> Answer:
    return Answer(status, {"detail": detail, "retry_after": seconds}, {"Retry-After": str(seconds)})


def create_app(
    settings: ApiSettings,
    *,
    runner: Runner | None = None,
    config: SourcesConfig | None = None,
    seed_sample: bool = False,
    fetch: Fetch | None = None,
    resolve: Resolve | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """The API. A run that names no feeds reads the sources in `config` (`serve --config`).
    With `seed_sample`, the committed sample draft is written at start, once (see
    `newsletter.api.seed`). Feeds are checked with `fetch` and `resolve` (default
    `http_get` and `resolve_host`); `clock` times the check's rate cap. Those three are
    for tests."""
    fetch = fetch or http_get
    resolve = resolve or resolve_host
    checks = RateLimit(CHECKS_PER_MINUTE, 60, clock)

    def check(url: str):
        return check_feed(url, fetch=fetch, resolve=resolve)

    redact = redactor(settings.secrets)
    drafts = DraftStore(settings.data_dir)
    runs = RunStore(settings.data_dir)
    jobs = Jobs(
        runner or default_runner(settings.env),
        runs,
        drafts,
        timeout=settings.run_timeout,
        secrets=settings.secrets,
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
            "its status, list, read, edit and approve drafts, and preview an approved "
            "draft as an HTML email. Nothing here sends mail. Every route but "
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
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Model-Preset",
            "X-Model-Key",
            "X-Demo-Code",
        ],
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

    @app.exception_handler(Answer)
    async def answer(request: Request, error: Answer) -> JSONResponse:
        return JSONResponse(redact(error.body), status_code=error.status, headers=error.headers)

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

    @app.get("/api/providers", response_model=list[ProviderView])
    def list_providers() -> list[dict[str, Any]]:
        """The model providers a run may use, from `providers.json`. Holds no secret.

        No token needed: the page asks for this list before it holds any credential.
        """
        return public_providers(PROVIDERS)

    @app.get("/api/presets", response_model=list[PresetView])
    def list_presets() -> list[dict[str, Any]]:
        """Sets of public feeds (AI news, science, tech) the run form can fill in.

        No token needed, like the provider list: it names public feeds only.
        """
        return PRESETS

    @app.post(
        "/api/feeds/check",
        response_model=FeedCheckView,
        responses={**WRITE_ERRORS, **TOO_MANY},
    )
    def check_one_feed(raw: Guarded) -> dict[str, Any]:
        """Whether a run can read this address: fetched as every feed is (robots.txt, the
        size cap, http and https only, never a private or loopback address), it answers,
        parses as a feed and has at least one item. No model call. At most 30 a minute,
        for every caller together; past that, 429 with `retry_after`."""
        body = parse(CheckBody, raw)
        if (wait := checks.take()) is not None:
            raise retry_later(429, "too many feed checks; wait and try again", wait)
        return asdict(check(body.url))

    @app.post(
        "/api/runs",
        status_code=202,
        response_model=RunView,
        response_model_exclude_unset=True,
        responses={
            202: {"description": "The run is queued; follow it at `/api/runs/{id}`."},
            **WRITE_ERRORS,
            **RUN_GOING,
        },
    )
    def start_run(
        raw: Guarded,
        credentials: Annotated[Credentials | None, Depends(request_credentials)],
    ) -> dict[str, Any]:
        """Start a run. With `X-Model-Preset` and `X-Model-Key` it runs on that provider
        with the visitor's key; with `X-Demo-Code`, on the demo gateway; with neither, on
        the server's own key. The key or code is used for this run's model calls only and
        is in nothing stored, returned or logged.

        Every feed the body names is checked first, as `POST /api/feeds/check` does: the
        run starts with the feeds that pass and lists the rest in its `Fetching feeds`
        step; when none pass it is refused with 422 and each feed's `errors`. Feeds from
        the sources config are the server's own and are not checked. In demo mode a run
        takes at most 6 feeds, reads at most 3 items of each, and ignores `date`."""
        body = parse(RunBody, raw)
        if settings.demo:
            if len(body.feeds) > DEMO_MAX_FEEDS:
                raise HTTPException(422, f"a demo run takes at most {DEMO_MAX_FEEDS} feeds")
            body = body.model_copy(update={"date": None})
        if credentials is not None:
            try:
                run_settings(settings.env, credentials)
            except ConfigError as error:
                raise HTTPException(503, str(error)) from None
        if jobs.busy:
            raise retry_later(409, BUSY, BUSY_RETRY_AFTER)
        request = replace(_run_request(body, config), credentials=credentials)
        if settings.demo:
            per_feed = min(DEMO_PER_FEED, request.per_feed or DEMO_PER_FEED)
            request = replace(request, per_feed=per_feed)
        if body.feeds:
            request = _checked(request, check)
        try:
            return jobs.start(request)
        except Busy:
            raise retry_later(409, BUSY, BUSY_RETRY_AFTER) from None

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

    @app.get(
        "/api/drafts/{draft_id}/email",
        dependencies=[Depends(guard)],
        response_model=EmailPreview,
        responses={
            **UNAUTHORIZED,
            **NOT_FOUND,
            409: {
                "description": "The draft's current text is not approved.",
                "model": ErrorBody,
            },
        },
    )
    def draft_email(draft_id: str) -> dict[str, str]:
        """The approved text as an HTML email, in the issue's shell. Nothing is sent."""
        draft = _found(drafts.get, draft_id)
        if not is_approved(draft):
            raise HTTPException(409, "the draft's current text is not approved; approve it first")
        return {"html": render_email(draft["text"])}

    return app


def _checked(request: RunRequest, check) -> RunRequest:
    """`request` with only the feeds that pass the check, and the rest as `failed_feeds`;
    422 with every feed's error when none pass."""
    with ThreadPoolExecutor(max_workers=min(8, len(request.feeds))) as pool:
        results = list(pool.map(check, request.feeds))
    failed = [
        FeedFailure(url, result.error or "")
        for url, result in zip(request.feeds, results, strict=True)
        if not result.ok
    ]
    if len(failed) == len(request.feeds):
        errors = [{"url": failure.url, "error": failure.reason} for failure in failed]
        raise Answer(422, {"detail": "no feed could be read", "errors": errors})
    good = [url for url, result in zip(request.feeds, results, strict=True) if result.ok]
    return replace(request, feeds=good, failed_feeds=failed)


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
    per_feed = body.per_feed or (config.per_feed if config else None)
    if body.feeds:
        return RunRequest(
            body.feeds, body.title, topic, body.date, per_feed=per_feed, focus=body.focus
        )
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
        focus=body.focus,
    )
