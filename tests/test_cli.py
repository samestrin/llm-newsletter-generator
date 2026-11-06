import json
import re
from html.parser import HTMLParser
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
    "NEWSLETTER_REASONING_MODEL": "reason-1",
    "NEWSLETTER_EMBEDDING_MODEL": "embed-1",
    "NEWSLETTER_MAX_RETRIES": "0",
}
PASSAGES = {
    "Scaling notes": "Notes on scaling",
    "A new open model": "The model is open",
    "Benchmarks revisited": "Benchmarks are hard",
    # The run from a sources config, at the end of this file.
    "City council approves new bike lanes": "12 km of protected bike lanes",
    "Acme releases Falcon-9 open-weight model": "Apache 2.0",
    "A mystery launch": "A mystery launch",
}
FALSE_CLAIM = "The model has one trillion parameters."


def fixture_web(url, headers, timeout):
    bodies = {RSS: "rss.xml", ATOM: "atom.xml"}
    if url not in bodies:
        return HttpResponse(404)
    return HttpResponse(200, {}, (FIXTURES / bodies[url]).read_bytes())


def apart(texts):
    """Embeddings that put no two items together and every item as close to the topic
    (the first text), so the stories keep the order they came in."""
    return [[1.0] + [3.0 if i == n else 0.0 for i in range(len(texts))] for n in range(len(texts))]


def model(
    *,
    fail_models=(),
    edit_down=False,
    rewrite=lambda text: f"In my voice: {text}",
    embed=apart,
):
    """A fake endpoint answering claims, checks and the edit by their schema name, a
    call with no schema (the voice pass) with `rewrite` of the text it was given, and an
    embedding call with `embed` of its inputs."""

    def reply(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            if body["model"] in fail_models:
                return httpx.Response(500, json={"error": "down"})
            vectors = embed(body["input"])
            data = [{"index": n, "embedding": v} for n, v in enumerate(vectors)]
            return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 3}})
        kind = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        prompt = body["messages"][-1]["content"]
        if body["model"] in fail_models or (edit_down and kind == "issue_edit"):
            return httpx.Response(500, json={"error": "down"})
        if kind is None:
            text = re.search(r"Text:\n\n(.*)\Z", prompt, re.S).group(1)
            content = rewrite(text)
        elif kind == "story_claims":
            title = re.search(r'story "(.*)"', prompt).group(1)
            claims = [{"text": f"{title}: 42% faster.", "passage": PASSAGES[title]}]
            if title == "A new open model":
                claims.append({"text": FALSE_CLAIM, "passage": "Second paragraph"})
                claims.append({"text": "An invented one.", "passage": "words not in the text"})
            content = json.dumps({"claims": claims})
        elif kind == "claim_check":
            content = json.dumps({"supported": FALSE_CLAIM not in prompt, "reason": "checked"})
        else:
            ids = re.findall(r"^- (s\d+\.c\d+):", prompt, re.M)
            stories = list(dict.fromkeys(i.split(".")[0] for i in ids))
            content = json.dumps(
                {
                    "lead": stories[0],
                    "order": stories,
                    "themes": [
                        {"name": "Open models", "claim_ids": ids[:1]},
                        {"name": "Speed", "claim_ids": ids},
                    ],
                    "intro": {"text": "This week, 2 labs.", "claim_ids": ids[:1]},
                    "why_it_matters": [
                        {
                            "story_id": sid,
                            "text": "It matters.",
                            "claim_ids": [i for i in ids if i.startswith(sid + ".")][:1],
                        }
                        for sid in stories
                    ],
                }
            )
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}
        message = {"content": content}
        return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage})

    return httpx.MockTransport(reply)


def legacy_model():
    """The plain writer behind `newsletter archive`: every call answers one summary."""

    def reply(request):
        body = json.loads(request.content)
        text = "In my voice." if body["model"] == "reason-1" else "A summary."
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    return httpx.MockTransport(reply)


def run(argv, env=ENV, transport=None, voice_transport=None):
    if transport is None:
        transport = model() if argv[0] == "issue" else legacy_model()
    return main(
        argv, env=env, fetch=fixture_web, llm_transport=transport, voice_transport=voice_transport
    )


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
        "YOURVOICE_API_KEY": "yv-key",
        "NEWSLETTER_VOICE_SAMPLES": str(samples),
        "NEWSLETTER_VOICE_CACHE_DIR": str(tmp_path / "cache"),
    }


def claim_lines(markdown):
    return [line for line in markdown.splitlines() if line.startswith(r"- \[")]


def test_issue_command_writes_the_edited_issue_with_every_claim_linked(tmp_path, capsys):
    out = tmp_path / "issue.md"
    code = run(["issue", RSS, ATOM, "--title", "AI Weekly", "-o", str(out)])

    markdown = out.read_text()
    assert markdown.startswith("# AI Weekly\n")
    for link in [
        "https://example.com/posts/open-model",
        "https://example.com/posts/benchmarks",
        "https://lab.example.org/scaling",
    ]:
        assert f"(<{link}>)" in markdown
    assert "This week, 2 labs." in markdown and "**Why it matters:** It matters." in markdown
    # The false claim stays visible and marked; the invented one is dropped and reported.
    assert f"**Flagged: unsupported, the source does not support this claim.** {FALSE_CLAIM}" in (
        markdown
    )
    assert "An invented one." in markdown.split("## Run report")[1]
    assert code == 1
    err = capsys.readouterr().err
    assert "3 stories" in err and "1 flagged claim" in err and "1 dropped claim" in err


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.append(dict(attrs).get("href"))


def test_issue_command_writes_the_html_email_with_every_claim_linked_and_flagged(tmp_path):
    out, html = tmp_path / "issue.md", tmp_path / "issue.html"
    run(["issue", RSS, "https://example.com/missing.xml", "-o", str(out), "--html", str(html)])

    page = html.read_text(encoding="utf-8")
    anchors = Anchors()
    anchors.feed(page)
    for link in ["https://example.com/posts/open-model", "https://example.com/posts/benchmarks"]:
        assert link in anchors.hrefs
    assert "<html" in page.lower()
    assert FALSE_CLAIM in page and "Flagged: unsupported" in page
    # The run report lists the failed feed and the cost per step.
    assert "https://example.com/missing.xml" in page and "claims-s1" in page


def test_issue_command_prints_to_stdout_and_exits_zero_when_all_is_well(capsys):
    code = run(["issue", ATOM, "--title", "AI Weekly", "--date", "2026-11-03"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("# AI Weekly\n\n*2026-11-03*")
    assert "(<https://lab.example.org/scaling>)" in captured.out
    assert "0 flagged claims" in captured.err


def test_the_report_lists_a_failed_feed_and_the_cost_per_step(capsys):
    code = run(["issue", ATOM, "https://example.com/missing.xml"])

    captured = capsys.readouterr()
    assert code == 1
    assert "HTTP 404" in captured.out and "1 feed problem" in captured.err
    for step in ["claims-s1", "check-s1.c1", "edit"]:
        assert f"| {step} |" in captured.out
    assert "$0.001000" in captured.out


def test_a_failed_editor_call_is_reported_and_the_story_still_listed(capsys):
    code = run(["issue", ATOM], transport=model(edit_down=True))

    captured = capsys.readouterr()
    assert code == 1
    assert "(<https://lab.example.org/scaling>)" in captured.out
    assert "editor failed" in captured.err


def test_issue_command_fails_when_a_model_call_fails(capsys):
    code = run(["issue", ATOM], transport=model(fail_models=("fast-1",)))

    captured = capsys.readouterr()
    assert code == 1
    assert "(<https://lab.example.org/scaling>)" in captured.out


def test_issue_command_needs_all_three_models_before_fetching_anything(capsys):
    fetched = []

    def fetch(url, headers, timeout):
        fetched.append(url)
        return HttpResponse(404)

    fast_only = {k: v for k, v in ENV.items() if k != "NEWSLETTER_REASONING_MODEL"}
    no_embedding = {k: v for k, v in ENV.items() if k != "NEWSLETTER_EMBEDDING_MODEL"}
    for env, name in [
        ({}, "NEWSLETTER_FAST_MODEL"),
        (fast_only, "NEWSLETTER_REASONING_MODEL"),
        (no_embedding, "NEWSLETTER_EMBEDDING_MODEL"),
    ]:
        assert main(["issue", ATOM], env=env, fetch=fetch, llm_transport=model()) == 2
        assert name in capsys.readouterr().err
    assert fetched == []


def test_issue_command_rejects_a_bad_setting(capsys):
    code = run(["issue", ATOM], env={**ENV, "NEWSLETTER_TIMEOUT": "soon"})

    assert code == 2
    assert "NEWSLETTER_TIMEOUT" in capsys.readouterr().err


def test_issue_command_uses_the_cache_dir(tmp_path):
    run(["issue", ATOM, "--cache-dir", str(tmp_path / "cache")])

    assert list((tmp_path / "cache").glob("*.xml"))


def test_issue_command_caps_the_items_per_feed(capsys):
    code = run(["issue", ATOM, RSS, "--per-feed", "1"])

    assert code == 1, "the RSS fixture's entry with no link is still reported"
    assert "2 stories" in capsys.readouterr().err


def test_issue_command_needs_a_feed():
    with pytest.raises(SystemExit) as exc:
        run(["issue"])
    assert exc.value.code == 2


def test_issue_command_is_neutral_without_a_voice_key(capsys):
    code = run(["issue", ATOM])

    out = capsys.readouterr().out
    assert code == 0
    assert "In my voice" not in out
    assert r"Neutral voice: YOURVOICE\_API\_KEY is not set." in out


def test_issue_command_writes_the_editors_prose_in_the_writers_voice(tmp_path, capsys):
    run(["issue", RSS, ATOM, "--date", "2026-11-03"])
    neutral = capsys.readouterr().out

    code = run(
        ["issue", RSS, ATOM, "--date", "2026-11-03"],
        env=voice_env(tmp_path),
        voice_transport=yourvoice(),
    )

    out = capsys.readouterr().out
    assert code == 1, "the flagged and dropped claims still fail the run"
    assert "In my voice: This week, 2 labs." in out
    assert "**Why it matters:** In my voice: It matters." in out
    assert "- In my voice: Open models [1](<https://example.com/posts/open-model>)" in out
    # Claims, their numbers and every source and citation link are byte for byte the same.
    assert claim_lines(out) == claim_lines(neutral) and len(claim_lines(out)) == 4
    assert re.findall(r"\(<[^>]+>\)", out) == re.findall(r"\(<[^>]+>\)", neutral)
    assert "Writer's voice: 6 of 6 texts rewritten." in out


def test_a_voice_rewrite_that_changes_a_number_keeps_the_neutral_prose(tmp_path, capsys):
    code = run(
        ["issue", ATOM],
        env=voice_env(tmp_path),
        transport=model(rewrite=lambda text: text.replace("2 labs", "3 labs")),
        voice_transport=yourvoice(),
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "This week, 2 labs." in out and "3 labs" not in out
    assert "`intro`: rewrite changed facts: number 2 was dropped" in out


def test_issue_command_keeps_neutral_prose_when_the_voice_fails(tmp_path, capsys):
    code = run(["issue", ATOM], env=voice_env(tmp_path), voice_transport=yourvoice(status=500))

    out = capsys.readouterr().out
    assert code == 1
    assert "This week, 2 labs." in out and "In my voice" not in out
    assert "Writer's voice: 0 of 4 texts rewritten." in out
    assert "`intro`: voice profile failed: HTTP 500" in out
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
        llm_transport=legacy_model(),
        voice_transport=yourvoice(),
    )

    (page,) = (site / "issues").glob("*/index.html")
    text = page.read_text(encoding="utf-8")
    assert code == 0
    assert "In my voice." in text and "Flagged: unchecked" in text


# A run from a sources config: a feed, a page and Hacker News, grouped and ranked.

RUN_FIXTURES = Path(__file__).parent / "fixtures" / "run"
NEWS = "https://news.example.com/feed.xml"
PAGE = "https://blog.example.net/2026/11/falcon-9"
HN = "https://hacker-news.firebaseio.com/v0"
HN_ITEMS = {
    1: {
        "type": "story",
        "id": 1,
        "title": "Acme ships Falcon-9 with weights anyone can download",
        "url": "https://ml.example.io/posts/falcon-9",
        "score": 300,
    },
    3: {
        "type": "story",
        "id": 3,
        "title": "A mystery launch",
        "score": 150,
        "url": "https://mystery.example.org/launch",
    },
}


def run_web(url, headers, timeout):
    if url == NEWS:
        return HttpResponse(200, {}, (RUN_FIXTURES / "feed.xml").read_bytes())
    if url == PAGE:
        body = (RUN_FIXTURES / "page.html").read_bytes()
        return HttpResponse(200, {"content-type": "text/html; charset=utf-8"}, body)
    if url == f"{HN}/topstories.json":
        return HttpResponse(200, {}, b"[1, 2, 3]")
    for story_id, story in HN_ITEMS.items():
        if url == f"{HN}/item/{story_id}.json":
            return HttpResponse(200, {}, json.dumps(story).encode())
    return HttpResponse(404)  # robots.txt and story 2


def by_meaning(texts):
    """The topic first; Falcon-9 is close to it, bike lanes far, the mystery launch has an
    embedding that cannot be compared."""

    def vector(text):
        if "Falcon-9" in text:
            return [0.9, 0.1, 0.42]
        if "bike lanes" in text:
            return [0.0, 1.0, 0.0]
        if "mystery" in text:
            return [0.0, 0.0, 0.0]
        return [1.0, 0.0, 0.3]

    return [vector(text) for text in texts]


def test_issue_from_a_config_groups_copies_ranks_by_topic_and_reports_what_failed(tmp_path, capsys):
    config = tmp_path / "sources.yaml"
    config.write_text(
        "topic: open-weight language models\n"
        f"sources:\n  - {NEWS}\n  - page: {PAGE}\n  - hackernews: {{min_score: 100}}\n",
        encoding="utf-8",
    )
    html = tmp_path / "issue.html"
    code = main(
        ["issue", "--config", str(config), "--html", str(html)],
        env=ENV,
        fetch=run_web,
        llm_transport=model(embed=by_meaning),
    )

    out, err = capsys.readouterr()
    issue, report = out.split("## Run report")
    stories = re.findall(r"^## (?:Lead story: )?\[(.*?)\]\(", issue, re.M)
    # Three copies of the Falcon-9 story are one story with three sources, ranked first,
    # and the bike lanes (far from the topic) come after it.
    assert "3 stories" in err
    assert stories == [
        "Acme releases Falcon-9 open-weight model",
        "City council approves new bike lanes",
        "A mystery launch",
    ]
    falcon = issue.split(stories[0], 1)[1].split("\n## ")[0]
    assert "Also covered by:" in falcon
    assert f"(<{PAGE}>)" in falcon and "(<https://ml.example.io/posts/falcon-9>)" in falcon
    # The failed Hacker News story and the item that could not be grouped are reported.
    assert f"{HN}/item/2.json" in report and "HTTP 404" in report
    assert "A mystery launch" in report and "ungrouped" in report
    page = html.read_text(encoding="utf-8").split("Run report")[1]
    assert "A mystery launch" in page and "ungrouped" in page
    assert code == 1
