import json
from datetime import date

import httpx

from newsletter.api.pipeline import RunRequest, run_issue, voice_edition
from newsletter.editor import Edition, Prose
from newsletter.evals.replay import ReplayClient
from newsletter.evals.run import load_cases
from newsletter.llm import Settings
from newsletter.sources import RssSource
from newsletter.sources.http import HttpResponse
from newsletter.voice import VoiceSettings

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


def replay(extra=None) -> ReplayClient:
    return ReplayClient(Settings(models=CASE.models), {**CASE.recorded, **(extra or {})})


def test_a_run_edits_and_renders_the_issue_with_its_flags_and_report():
    content = run_issue(request(), replay(), source=RssSource(fetch=web), voice=VoiceSettings())

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
        source=RssSource(fetch=web),
        voice=VoiceSettings(),
    )
    assert [f["url"] for f in content["report"]["failed_sources"]] == [missing]
    assert content["report"]["ok"] is False


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
        source=RssSource(fetch=web),
        voice=voice_settings(tmp_path),
        voice_transport=httpx.MockTransport(fake_yourvoice),
    )
    voice = content["report"]["voice"]
    assert voice["enabled"] is True
    assert "intro" in voice["voiced"]
    # Every other text was not recorded, so its rewrite failed and it stayed neutral.
    assert voice["kept_neutral"]
    assert all(k["id"].startswith("why-") for k in voice["kept_neutral"])
    assert "Three things happened this week." in content["text"]
    assert "Three developments this week." not in content["text"]
    assert "Writer's voice:" in content["text"]
    assert VOICE_KEY not in json.dumps(content)
