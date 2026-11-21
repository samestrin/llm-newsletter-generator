import json
from datetime import date

import httpx
import pytest

from newsletter.api.pipeline import RunRequest, run_issue, voice_edition
from newsletter.config import HackerNewsConfig
from newsletter.editor import Edition, Prose
from newsletter.evals.replay import ReplayClient
from newsletter.evals.run import load_cases
from newsletter.grouping import LeftOut
from newsletter.llm import ConfigError, LLMClient, Settings
from newsletter.render import RunReport
from newsletter.sources.http import HttpResponse
from newsletter.voice import VoiceSettings
from test_cli import ENV, HN, NEWS, PAGE, apart, by_meaning, model, run_web

CASE = next(case for case in load_cases() if case.name == "open-models-week")
VOICE_KEY = "yv-secret-key-123"


def web(url, headers, timeout):
    if url == CASE.feed_url:
        return HttpResponse(200, {}, CASE.feed)
    return HttpResponse(404)


def request(**changes) -> RunRequest:
    fields = {
        "feeds": [CASE.feed_url],
        "title": "Open models week",
        "topic": "AI models",
        "issue_date": date(2026, 11, 2),
    }
    return RunRequest(**{**fields, **changes})


class Replay(ReplayClient):
    """The recorded case, with every item kept apart by the grouping step."""

    def embed(self, step, inputs):
        return apart(inputs)


def replay(extra=None) -> ReplayClient:
    return Replay(Settings(models=CASE.models), {**CASE.recorded, **(extra or {})})


def test_a_run_edits_and_renders_the_issue_with_its_flags_and_report():
    content = run_issue(request(), replay(), fetch=web, voice=VoiceSettings())

    assert content["title"] == "Open models week"
    assert content["issue_date"] == "2026-11-02"
    assert content["text"].startswith("# Open models week")
    assert "## Run report" in content["text"]
    flagged = {flag["id"] for flag in content["flags"]}
    assert set(CASE.planted_false) <= flagged
    assert content["report"]["flagged_claims"] == content["flags"]
    assert content["report"]["voice"]["skipped"] == "YOURVOICE_API_KEY is not set"
    steps = [step["step"] for step in content["report"]["steps"]]
    assert steps[0] == "claims-s1" and steps[-1] == "edit"
    json.dumps(content)  # Stored as JSON as it is.


def test_a_feed_that_fails_is_in_the_report():
    missing = "https://gone.example.org/feed.xml"
    content = run_issue(
        request(feeds=[CASE.feed_url, missing]),
        replay(),
        fetch=web,
        voice=VoiceSettings(),
    )
    assert [f["url"] for f in content["report"]["failed_sources"]] == [missing]
    assert content["report"]["ok"] is False


def test_a_run_reads_feeds_pages_and_hacker_news_and_groups_them():
    llm = LLMClient(Settings.from_env(ENV), transport=model(embed=by_meaning))
    run = request(
        feeds=[NEWS],
        pages=[PAGE],
        hackernews=HackerNewsConfig(min_score=100),
        topic="open-weight language models",
    )
    content = run_issue(run, llm, fetch=run_web, voice=VoiceSettings())

    report = content["report"]
    assert [f["url"] for f in report["failed_sources"]] == [f"{HN}/item/2.json"]
    assert [entry["title"] for entry in report["left_out"]] == ["A mystery launch"]
    assert [(a["link"], a["status"]) for a in report["linked_articles"]] == [
        ("https://ml.example.io/posts/falcon-9", "fetched"),
        ("https://mystery.example.org/launch", "fetched"),
    ]
    assert report["left_out"][0]["reason"].startswith("ungrouped:")
    assert report["grouping_error"] is None
    assert report["ok"] is False
    assert f"(<{PAGE}>)" in content["text"] and "Also covered by:" in content["text"]
    assert [s["step"] for s in report["steps"]][0] == "group"


def test_a_failed_embedding_call_is_reported_and_every_item_listed():
    llm = LLMClient(Settings.from_env(ENV), transport=model(fail_models=("embed-1",)))
    content = run_issue(request(feeds=[NEWS]), llm, fetch=run_web, voice=VoiceSettings())

    report = content["report"]
    assert report["grouping_error"].startswith("group:")
    assert len(report["left_out"]) == 2
    assert report["failed_sources"] == []
    assert report["ok"] is False
    assert "Grouping failed" in content["text"]


@pytest.mark.parametrize(
    "report",
    [
        RunReport(left_out=[LeftOut("A story", "https://a.test/s", "ungrouped: no vector")]),
        RunReport(grouping_error="group: HTTP 500"),
    ],
)
def test_an_item_left_out_or_a_failed_grouping_alone_makes_the_report_not_ok(report):
    assert report.ok is False
    assert report.to_dict()["ok"] is False


def fake_yourvoice(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["texts"]
    return httpx.Response(
        200, json={"document_count": len(texts), "results": [{"style_prompt": "Dry."}] * len(texts)}
    )


def voice_settings(tmp_path) -> VoiceSettings:
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "one.md").write_text("I write short sentences.", "utf-8")
    return VoiceSettings(api_key=VOICE_KEY, samples_dir=samples, cache_dir=tmp_path / "cache")


def edition() -> Edition:
    return Edition(
        stories=[],
        intro=Prose("Three labs shipped 3 models [s1.c1].", ["s1.c1"]),
        why_it_matters={"s1": Prose("It runs on 8 GB [s1.c2].", ["s1.c2"])},
    )


def test_voice_rewrites_the_editor_prose_and_keeps_a_rewrite_that_changed_a_fact_neutral(
    tmp_path,
):
    llm = replay(
        {
            "voice-intro": {"content": "So, three labs shipped 3 models [s1.c1]."},
            "voice-why-s1": {"content": "It runs on 16 GB [s1.c2]."},
        }
    )
    edited = edition()
    run = voice_edition(
        edited, llm, voice_settings(tmp_path), transport=httpx.MockTransport(fake_yourvoice)
    )
    assert edited.intro == Prose("So, three labs shipped 3 models [s1.c1].", ["s1.c1"])
    assert edited.why_it_matters["s1"] == Prose("It runs on 8 GB [s1.c2].", ["s1.c2"])
    assert run.voiced == ["intro"]
    assert [k.id for k in run.kept_neutral] == ["why-s1"]


def test_voice_without_editor_prose_has_nothing_to_rewrite(tmp_path):
    empty = Edition(stories=[], error="no supported claims to edit from")
    run = voice_edition(empty, replay(), voice_settings(tmp_path))
    assert run.voiced == [] and run.kept_neutral == []


def test_the_voice_pass_runs_between_the_editor_and_render(tmp_path):
    assert CASE.recorded["edit"]["content"]["intro"]["text"] == "Three developments this week."
    llm = replay({"voice-intro": {"content": "Three things happened this week."}})
    content = run_issue(
        request(),
        llm,
        fetch=web,
        voice=voice_settings(tmp_path),
        voice_transport=httpx.MockTransport(fake_yourvoice),
    )
    voice = content["report"]["voice"]
    assert voice["enabled"] is True
    assert "intro" in voice["voiced"]
    # Every other text was not recorded, so its rewrite failed and it stayed neutral.
    assert voice["kept_neutral"]
    assert all(k["id"].startswith(("theme-", "why-")) for k in voice["kept_neutral"])
    assert "Three things happened this week." in content["text"]
    assert "Three developments this week." not in content["text"]
    assert "Writer's voice:" in content["text"]
    assert VOICE_KEY not in json.dumps(content)


def test_the_default_runner_reads_its_settings_from_the_environment():
    from newsletter.api.pipeline import default_runner

    run = default_runner({"NEWSLETTER_MAX_RETRIES": "0"})
    # Nothing listens on port 9, so the feed fails without leaving this machine.
    content = run(request(feeds=["http://127.0.0.1:9/feed.xml"]))
    assert [f["url"] for f in content["report"]["failed_sources"]] == [
        "http://127.0.0.1:9/feed.xml"
    ]
    assert content["report"]["editor_error"] == "no supported claims to edit from"
    assert content["report"]["voice"]["skipped"] == "YOURVOICE_API_KEY is not set"


def test_a_run_with_items_and_no_embedding_model_fails_with_the_setting_named():
    env = {k: v for k, v in ENV.items() if k != "NEWSLETTER_EMBEDDING_MODEL"}
    llm = LLMClient(Settings.from_env(env), transport=model())
    with pytest.raises(ConfigError, match="NEWSLETTER_EMBEDDING_MODEL"):
        run_issue(request(feeds=[NEWS]), llm, fetch=run_web, voice=VoiceSettings())


# Live progress: the start and end of each stage, reported to an optional callback.

STAGES = ["feeds", "articles", "grouping", "claims", "check", "editor"]


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, step, state, *, detail=None, done=None, total=None):
        self.events.append((step, state, detail))


def test_a_run_reports_the_start_and_end_of_each_stage_in_order():
    progress = Recorder()
    run_issue(request(), replay(), fetch=web, voice=VoiceSettings(), progress=progress)
    # A stage counting as it goes reports `running` again with each count: one entry each.
    states = []
    for step, state, _ in progress.events:
        if not states or states[-1] != (step, state):
            states.append((step, state))
    expected = [(stage, state) for stage in STAGES for state in ("running", "done")]
    assert states == [*expected, ("done", "running")]
    assert all(isinstance(detail, str | None) for _, _, detail in progress.events)


def test_a_run_whose_editor_fails_reports_the_editor_as_failed():
    recorded = {k: v for k, v in CASE.recorded.items() if not k.startswith("edit")}
    llm = Replay(Settings(models=CASE.models), recorded)
    progress = Recorder()
    content = run_issue(request(), llm, fetch=web, voice=VoiceSettings(), progress=progress)
    assert content["report"]["editor_error"]
    editor = [state for step, state, _ in progress.events if step == "editor"]
    assert editor == ["running", "failed"]


def test_progress_details_hold_counts_and_no_model_text():
    marker = "MODEL-WROTE-THIS"
    progress = Recorder()
    llm = replay({"edit": {"content": {**CASE.recorded["edit"]["content"], "lead": marker}}})
    run_issue(request(), llm, fetch=web, voice=VoiceSettings(), progress=progress)
    details = [detail for _, _, detail in progress.events if detail]
    assert details, "every finished stage says what it did"
    for detail in details:
        assert marker not in detail
        assert any(ch.isdigit() for ch in detail)


def test_a_run_without_a_progress_callback_is_unchanged():
    without = run_issue(request(), replay(), fetch=web, voice=VoiceSettings())
    with_progress = run_issue(
        request(), replay(), fetch=web, voice=VoiceSettings(), progress=Recorder()
    )
    assert with_progress == without


def test_a_failed_embedding_call_reports_the_grouping_as_failed():
    llm = LLMClient(Settings.from_env(ENV), transport=model(fail_models=("embed-1",)))
    progress = Recorder()
    content = run_issue(
        request(feeds=[NEWS]), llm, fetch=run_web, voice=VoiceSettings(), progress=progress
    )
    assert content["report"]["grouping_error"]
    grouping = [state for step, state, _ in progress.events if step == "grouping"]
    assert grouping == ["running", "failed"]


# Live progress: the counts inside each stage, `done` of `total` and a detail as it goes.

GONE = "https://gone.example.org/feed.xml"
MYSTERY = "https://mystery.example.org/launch"


class Counts:
    def __init__(self):
        self.events = []

    def __call__(self, step, state, *, detail=None, done=None, total=None):
        self.events.append(
            {"step": step, "state": state, "detail": detail, "done": done, "total": total}
        )

    def of(self, step):
        return [event for event in self.events if event["step"] == step]


def mystery_gone(url, headers, timeout):
    """The run's fake web, with the article one Hacker News story links to gone."""
    return HttpResponse(404) if url == MYSTERY else run_web(url, headers, timeout)


def counted_run(progress=None) -> dict:
    llm = LLMClient(Settings.from_env(ENV), transport=model(embed=by_meaning))
    run = request(
        feeds=[NEWS, GONE],
        pages=[PAGE],
        hackernews=HackerNewsConfig(min_score=100),
        topic="open-weight language models",
    )
    extra = {} if progress is None else {"progress": progress}
    return run_issue(run, llm, fetch=mystery_gone, voice=VoiceSettings(), **extra)


def test_each_stage_ends_with_counts_equal_to_the_real_totals():
    progress = Counts()
    report = counted_run(progress)["report"]
    last = {event["step"]: event for event in progress.events}
    calls = [step["step"] for step in report["steps"]]

    # Two feeds, one page and Hacker News: four sources, the gone feed failed.
    (gone,) = [f for f in report["failed_sources"] if f["url"] == GONE]
    assert (last["feeds"]["done"], last["feeds"]["total"]) == (4, 4)
    assert last["feeds"]["detail"] == f"3 of 4 loaded, 1 failed: {GONE} {gone['reason']}"

    # Two Hacker News stories link out; the mystery article could not be read.
    statuses = [a["status"] for a in report["linked_articles"]]
    assert statuses == ["fetched", "skipped"]
    assert (last["articles"]["done"], last["articles"]["total"]) == (2, 2)
    assert last["articles"]["detail"] == (
        "1 of 2 fetched, 1 left out: the linked article could not be read"
    )

    stories = sum(1 for call in calls if call.startswith("claims-"))
    checks = sum(1 for call in calls if call.startswith("check-"))
    assert stories and checks
    assert (last["claims"]["done"], last["claims"]["total"]) == (stories, stories)
    assert (last["check"]["done"], last["check"]["total"]) == (checks, checks)
    assert f"story {stories} of {stories}" in [e["detail"] for e in progress.of("claims")]
    assert f"{checks} of {checks}" in [e["detail"] for e in progress.of("check")]


def test_counts_only_go_up_and_each_total_never_changes():
    progress = Counts()
    counted_run(progress)
    for step in ("feeds", "articles", "claims", "check"):
        events = progress.of(step)
        dones = [e["done"] for e in events if e["done"] is not None]
        totals = {e["total"] for e in events if e["total"] is not None}
        assert len(dones) > 2, step  # Counted as it goes, not only at the end.
        assert dones == sorted(dones), step
        assert dones[0] == 0, step
        assert len(totals) == 1 and dones[-1] == totals.pop(), step


def test_a_stage_with_nothing_to_count_says_so():
    progress = Counts()
    run_issue(request(), replay(), fetch=web, voice=VoiceSettings(), progress=progress)
    articles = progress.of("articles")[-1]
    assert (articles["done"], articles["total"]) == (0, 0)
    assert articles["detail"] == "0 of 0 fetched"
    feeds = progress.of("feeds")[-1]
    assert feeds["detail"] == "1 of 1 loaded"


def test_a_counted_run_without_a_progress_callback_is_unchanged():
    def timeless(content):  # A model call's `seconds` is wall-clock time.
        for step in content["report"]["steps"]:
            step.pop("seconds")
        return content

    assert timeless(counted_run(Counts())) == timeless(counted_run())


# A failed editor: one plain sentence built in code, the raw error in the report only.

EDITOR_TIMED_OUT = (
    "The editor did not finish in time. Your sources were read and checked; "
    "try again, or edit the draft by hand."
)


def editor_answers(fail):
    """The fake model of `test_cli`, with its edit call replaced by `fail(request)`."""
    inner = model()

    def reply(request):
        body = json.loads(request.content)
        if (body.get("response_format") or {}).get("json_schema", {}).get("name") == "issue_edit":
            return fail(request)
        return inner.handle_request(request)

    return httpx.MockTransport(reply)


def timed_out(request):
    raise httpx.ReadTimeout("read timed out", request=request)


def test_a_timed_out_editor_gives_the_plain_sentence_and_the_raw_error_in_the_report():
    llm = LLMClient(Settings.from_env(ENV), transport=editor_answers(timed_out))
    content = run_issue(request(feeds=[NEWS]), llm, fetch=run_web, voice=VoiceSettings())

    report = content["report"]
    assert report["editor_message"] == EDITOR_TIMED_OUT
    assert report["editor_error"] == "edit: timed out after 300.0s"
    assert report["ok"] is False


def test_an_editor_that_fails_otherwise_gives_a_plain_sentence_without_the_error():
    down = editor_answers(lambda request: httpx.Response(500, json={"error": "down"}))
    llm = LLMClient(Settings.from_env(ENV), transport=down)
    content = run_issue(request(feeds=[NEWS]), llm, fetch=run_web, voice=VoiceSettings())

    report = content["report"]
    assert report["editor_error"].startswith("edit: HTTP 500")
    assert report["editor_message"] == (
        "The editor did not finish. Your sources were read and checked; "
        "try again, or edit the draft by hand."
    )


def test_an_editor_with_nothing_to_edit_from_says_so_plainly():
    llm = Replay(Settings(models=CASE.models), {})  # No claims: every claim call fails.
    content = run_issue(request(), llm, fetch=web, voice=VoiceSettings())

    assert content["report"]["editor_error"] == "no supported claims to edit from"
    assert content["report"]["editor_message"] == (
        "The editor had no supported claims to write from. Your sources were read and "
        "checked; edit the draft by hand."
    )


def test_an_editor_that_finishes_has_no_message():
    content = run_issue(request(), replay(), fetch=web, voice=VoiceSettings())
    assert content["report"]["editor_error"] is None
    assert content["report"]["editor_message"] is None
