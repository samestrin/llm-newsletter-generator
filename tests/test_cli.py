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
