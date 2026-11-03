import json
from pathlib import Path

import httpx
import pytest

from newsletter.cli import build_parser, main
from newsletter.sources.http import HttpResponse


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "newsletter" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert "newsletter 0.0.4" in capsys.readouterr().out


def test_main_returns_zero():
    assert main([]) == 0


FIXTURES = Path(__file__).parent / "fixtures" / "feeds"
RSS = "https://example.com/feed.xml"
ATOM = "https://lab.example.org/atom.xml"
ENV = {
    "NEWSLETTER_BASE_URL": "http://model.test/v1",
    "NEWSLETTER_FAST_MODEL": "fast-1",
    "NEWSLETTER_MAX_RETRIES": "0",
}


def fixture_web(url, headers, timeout):
    bodies = {RSS: "rss.xml", ATOM: "atom.xml"}
    if url not in bodies:
        return HttpResponse(404)
    return HttpResponse(200, {}, (FIXTURES / bodies[url]).read_bytes())


def model(status=200):
    def reply(request):
        if status != 200:
            return httpx.Response(status, json={"error": "down"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "A summary."}}]})

    return httpx.MockTransport(reply)


def run(argv, env=ENV, transport=None):
    return main(argv, env=env, fetch=fixture_web, llm_transport=transport or model())


def test_issue_command_writes_markdown_with_every_item_linked(tmp_path, capsys):
    out = tmp_path / "issue.md"
    code = run(["issue", RSS, ATOM, "--title", "AI Weekly", "--topic", "AI", "-o", str(out)])

    markdown = out.read_text()
    assert markdown.startswith("# AI Weekly\n")
    for link in [
        "https://example.com/posts/open-model",
        "https://example.com/posts/benchmarks",
        "https://lab.example.org/scaling",
    ]:
        assert f"(<{link}>)" in markdown
    # The RSS fixture has an entry with no link, which the run report lists.
    assert code == 1
    assert "3 stories" in capsys.readouterr().err


def test_issue_command_prints_to_stdout_and_exits_zero_when_all_is_well(capsys):
    code = run(["issue", ATOM, "--title", "AI Weekly", "--date", "2026-11-03"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("# AI Weekly\n\n*2026-11-03*")
    assert "(<https://lab.example.org/scaling>)" in captured.out


def test_issue_command_fails_when_a_model_call_fails(capsys):
    code = run(["issue", ATOM], transport=model(status=500))

    captured = capsys.readouterr()
    assert code == 1
    assert "(<https://lab.example.org/scaling>)" in captured.out
    assert "1 failed summary" in captured.err


def test_issue_command_fails_when_a_feed_fails(capsys):
    code = run(["issue", "https://example.com/missing.xml"])

    captured = capsys.readouterr()
    assert code == 1
    assert "HTTP 404" in captured.out
    assert "1 feed problem" in captured.err


def test_issue_command_needs_a_fast_model_before_fetching_anything(capsys):
    fetched = []

    def fetch(url, headers, timeout):
        fetched.append(url)
        return HttpResponse(404)

    code = main(["issue", ATOM], env={}, fetch=fetch, llm_transport=model())

    assert code == 2
    assert fetched == []
    assert "NEWSLETTER_FAST_MODEL" in capsys.readouterr().err


def test_issue_command_rejects_a_bad_setting(capsys):
    code = run(["issue", ATOM], env={**ENV, "NEWSLETTER_TIMEOUT": "soon"})

    assert code == 2
    assert "NEWSLETTER_TIMEOUT" in capsys.readouterr().err


def test_issue_command_uses_the_cache_dir(tmp_path):
    run(["issue", ATOM, "--cache-dir", str(tmp_path / "cache")])

    assert list((tmp_path / "cache").glob("*.xml"))


def test_issue_command_needs_a_feed():
    with pytest.raises(SystemExit) as exc:
        run(["issue"])
    assert exc.value.code == 2


def voiced_model():
    """The fast model summarizes; the reasoning model answers in the writer's voice."""

    def reply(request):
        body = json.loads(request.content)
        text = "In my voice." if body["model"] == "reason-1" else "A summary."
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    return httpx.MockTransport(reply)


def yourvoice(status=200):
    def reply(request):
        if status != 200:
            return httpx.Response(status, json={"error": "down"})
        texts = json.loads(request.content)["texts"]
        return httpx.Response(200, json={"results": [{"style_prompt": "Terse."} for _ in texts]})

    return httpx.MockTransport(reply)


def voice_env(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "post.md").write_text("My own words.", encoding="utf-8")
    return {
        **ENV,
        "NEWSLETTER_REASONING_MODEL": "reason-1",
        "YOURVOICE_API_KEY": "yv-key",
        "NEWSLETTER_VOICE_SAMPLES": str(samples),
        "NEWSLETTER_VOICE_CACHE_DIR": str(tmp_path / "cache"),
    }


def test_issue_command_is_neutral_without_a_voice_key(capsys):
    code = run(["issue", ATOM])

    assert code == 0
    assert r"Neutral voice: YOURVOICE\_API\_KEY is not set." in capsys.readouterr().out


def test_issue_command_writes_in_the_writers_voice_when_a_key_is_set(tmp_path, capsys):
    code = main(
        ["issue", ATOM],
        env=voice_env(tmp_path),
        fetch=fixture_web,
        llm_transport=voiced_model(),
        voice_transport=yourvoice(),
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "In my voice." in out and "A summary." not in out.split("## Sources")[0]
    assert "Writer's voice: 1 of 1 summaries rewritten." in out


def test_issue_command_fails_when_the_voice_fails(tmp_path, capsys):
    code = main(
        ["issue", ATOM],
        env=voice_env(tmp_path),
        fetch=fixture_web,
        llm_transport=voiced_model(),
        voice_transport=yourvoice(status=500),
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "A summary." in out
    assert "voice profile failed: HTTP 500" in out
    assert "yv-key" not in out


def test_issue_command_rejects_a_bad_voice_setting(capsys):
    code = run(["issue", ATOM], env={**ENV, "YOURVOICE_TIMEOUT": "soon"})

    assert code == 2
    assert "YOURVOICE_TIMEOUT" in capsys.readouterr().err


def test_archive_command_builds_a_labeled_demo_site(tmp_path, capsys):
    site = tmp_path / "site"
    code = run(["archive", ATOM, "--demo", "--site", str(site), "--date", "2026-11-03"])

    assert code == 0
    page = (site / "issues/2026-11-03/index.html").read_text(encoding="utf-8")
    assert "Automated demo" in page and "Automated demo" in (site / "index.html").read_text()
    assert "Flagged: unchecked" in page
    assert f"wrote {site}" in capsys.readouterr().err


def test_archive_command_refuses_to_build_without_demo_mode(tmp_path, capsys):
    fetched = []

    def fetch(url, headers, timeout):
        fetched.append(url)
        return HttpResponse(404)

    code = main(["archive", ATOM, "--site", str(tmp_path / "site")], env=ENV, fetch=fetch)

    assert code == 2
    assert fetched == []
    assert not (tmp_path / "site").exists()
    assert "--demo" in capsys.readouterr().err


def test_archive_command_builds_the_site_but_fails_when_a_feed_fails(tmp_path, capsys):
    site = tmp_path / "site"
    code = run(["archive", ATOM, "https://example.com/missing.xml", "--demo", "--site", str(site)])

    assert code == 1
    assert (site / "index.html").exists()
    assert "1 feed problem" in capsys.readouterr().err


def test_archive_command_needs_a_fast_model_before_fetching_anything(tmp_path, capsys):
    code = main(["archive", ATOM, "--demo", "--site", str(tmp_path / "s")], env={})

    assert code == 2
    assert not (tmp_path / "s").exists()
    assert "NEWSLETTER_FAST_MODEL" in capsys.readouterr().err


def test_archive_command_needs_a_site_dir():
    with pytest.raises(SystemExit) as exc:
        run(["archive", ATOM, "--demo"])
    assert exc.value.code == 2


def test_archive_command_caps_the_items_per_feed(tmp_path):
    site = tmp_path / "site"
    code = run(["archive", ATOM, RSS, "--demo", "--site", str(site), "--per-feed", "1"])

    (page,) = (site / "issues").glob("*/index.html")
    assert page.read_text(encoding="utf-8").count("<article>") == 2
    assert code == 1, "the RSS fixture's entry with no link is still reported"


@pytest.mark.parametrize("value", ["0", "-2", "many"])
def test_per_feed_must_be_a_positive_number(value):
    with pytest.raises(SystemExit) as exc:
        run(["issue", ATOM, "--per-feed", value])
    assert exc.value.code == 2


def test_archive_command_drafts_the_same_voiced_issue_as_the_issue_command(tmp_path):
    site = tmp_path / "site"
    code = main(
        ["archive", ATOM, "--demo", "--site", str(site)],
        env=voice_env(tmp_path),
        fetch=fixture_web,
        llm_transport=voiced_model(),
        voice_transport=yourvoice(),
    )

    (page,) = (site / "issues").glob("*/index.html")
    text = page.read_text(encoding="utf-8")
    assert code == 0
    assert "In my voice." in text and "Flagged: unchecked" in text
