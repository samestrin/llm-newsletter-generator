"""`newsletter archive --demo --markdown PATH --html PATH`: the issue Markdown and the HTML
email written from the same draft that builds the site, offline, with fake feeds and a
fake model whose errors carry a key, a file path and the model URL."""

import json
import re
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx

from newsletter.archive import render_email, render_markdown
from newsletter.cli import main
from newsletter.issue import Issue, Story
from newsletter.render import FLAG_LABELS
from newsletter.sources import FeedFailure, Item
from newsletter.sources.http import HttpResponse

ATOM = (Path(__file__).parent / "fixtures" / "feeds" / "atom.xml").read_bytes()
FEED = "https://lab.example.org/atom.xml"
API_KEY = "sk-test-SECRET-files-0123456789"
KEYED_FEED = f"https://feeds.example.net/rss?key={API_KEY}"
SECRET_PATH = "/home/runner/.cache/newsletter/feeds/secret.json"
MODEL_HOST = "model.internal"
MODEL_URL = f"http://{MODEL_HOST}:8080/v1"
ERROR_BODY = f'{{"error": "bad request, key {API_KEY}, see {SECRET_PATH}"}}'
ENV = {
    "NEWSLETTER_BASE_URL": MODEL_URL,
    "NEWSLETTER_API_KEY": API_KEY,
    "NEWSLETTER_FAST_MODEL": "fast-1",
    "NEWSLETTER_MAX_RETRIES": "0",
}
SECRETS = (API_KEY, "SECRET", SECRET_PATH, "secret.json", MODEL_HOST, "bad request")
MARKUP = '<script>alert("x")</script><b>Bold</b> & co'
UNCHECKED = FLAG_LABELS["unchecked"]


def fetch(url, headers, timeout):
    if url == FEED:
        return HttpResponse(200, {}, ATOM)
    if url == KEYED_FEED:
        return HttpResponse(500, {}, ERROR_BODY.encode())
    return HttpResponse(404)


def numbered_model(calls, *, fail_first=False):
    """Every call answers a new summary, so a second draft could not give the same text."""

    def reply(request):
        calls.append(json.loads(request.content))
        if fail_first and len(calls) == 1:
            return httpx.Response(400, json={"error": ERROR_BODY})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": f"Summary number {len(calls)}."}}]}
        )

    return httpx.MockTransport(reply)


def archive(tmp_path, *extra, feeds=(FEED,), fail_first=False):
    calls = []
    out = {"site": tmp_path / "site", "md": tmp_path / "issue.md", "html": tmp_path / "email.html"}
    argv = [
        "archive",
        *feeds,
        "--demo",
        "--date",
        "2026-11-03",
        "--site",
        str(out["site"]),
        "--markdown",
        str(out["md"]),
        "--html",
        str(out["html"]),
        *extra,
    ]
    code = main(argv, env=ENV, fetch=fetch, llm_transport=numbered_model(calls, fail_first=fail_first))
    return code, calls, out


def summaries(text):
    return re.findall(r"Summary number \d+\.", text)


def test_the_flags_write_the_markdown_and_the_email_from_the_draft_that_built_the_site(
    tmp_path, capsys
):
    code, calls, out = archive(tmp_path)

    assert code == 0
    page = (out["site"] / "issues/2026-11-03/index.html").read_text(encoding="utf-8")
    markdown = out["md"].read_text(encoding="utf-8")
    email = out["html"].read_text(encoding="utf-8")
    drafted = summaries(page)
    assert drafted and len(drafted) == len(calls), "one model call per story, one draft"
    assert summaries(markdown) == drafted
    assert summaries(email) == drafted
    err = capsys.readouterr().err
    assert f"wrote {out['site']}" in err
    assert f"wrote {out['md']}" in err and f"wrote {out['html']}" in err


def test_markdown_and_email_carry_the_demo_label_before_the_title(tmp_path):
    _, _, out = archive(tmp_path)

    markdown = out["md"].read_text(encoding="utf-8")
    assert markdown.index("Automated demo") < markdown.index("\n# ")
    assert "no human review" in markdown
    email = out["html"].read_text(encoding="utf-8")
    body = email[email.index("<body>") :]
    assert body.index("Automated demo") < body.index("<h1>")
    assert "no human review" in body


def test_every_summary_stays_flagged_in_the_markdown_and_the_email(tmp_path):
    _, _, out = archive(tmp_path)

    markdown = out["md"].read_text(encoding="utf-8")
    email = out["html"].read_text(encoding="utf-8")
    for text in (markdown, email):
        drafted = summaries(text)
        assert drafted
        for summary in drafted:
            before = text[: text.index(summary)]
            story = before[before.rindex("## ") if text is markdown else before.rindex("<h2>") :]
            assert UNCHECKED in story, summary


def test_the_email_has_no_archive_link_and_loads_nothing_from_outside(tmp_path):
    _, _, out = archive(tmp_path)

    parser = Tags()
    parser.feed(out["html"].read_text(encoding="utf-8"))
    assert "../../index.html" not in parser.hrefs
    assert not parser.sources, "no image, script or stylesheet to fetch"
    assert {"script", "img", "link", "iframe"}.isdisjoint(parser.tags)


def test_the_run_report_shows_no_key_file_path_or_model_url(tmp_path, capsys):
    code, _, out = archive(tmp_path, feeds=(FEED, KEYED_FEED), fail_first=True)

    assert code == 1, "a failed feed and a failed summary still fail the run"
    for path in [out["md"], out["html"], *sorted(out["site"].rglob("*.html"))]:
        text = path.read_text(encoding="utf-8")
        assert "Run report" in text
        assert "feeds.example.net: 1 problem" in text
        assert "lab.example.org: 1 problem" in text, "the failed summary"
        for secret in SECRETS:
            assert secret not in text, (path, secret)
        assert str(tmp_path) not in text, path
    err = capsys.readouterr().err
    for secret in (API_KEY, MODEL_HOST):
        assert secret not in err


def test_without_the_flags_only_the_site_is_written(tmp_path):
    calls = []
    site = tmp_path / "site"
    code = main(
        ["archive", FEED, "--demo", "--site", str(site)],
        env=ENV,
        fetch=fetch,
        llm_transport=numbered_model(calls),
    )

    assert code == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["site"]


def test_without_demo_mode_nothing_is_written(tmp_path, capsys):
    calls = []
    md, html = tmp_path / "issue.md", tmp_path / "email.html"
    code = main(
        ["archive", FEED, "--site", str(tmp_path / "s"), "--markdown", str(md), "--html", str(html)],
        env=ENV,
        fetch=fetch,
        llm_transport=numbered_model(calls),
    )

    assert code == 2
    assert calls == [] and list(tmp_path.iterdir()) == []
    assert "--demo" in capsys.readouterr().err


# The renderers, on an issue whose feed and model text is hostile.


def item(title="A story", link="https://example.com/a", feed="https://example.com/feed.xml"):
    return Item(
        title=title,
        link=link,
        summary="Some text.",
        published=datetime(2026, 11, 2, 9, 0, tzinfo=UTC),
        feed_url=feed,
        feed_title="Example News",
        rights="CC BY 4.0",
    )


def hostile_issue():
    return Issue(
        title=f"{MARKUP} ../../etc/passwd",
        topic="AI",
        issue_date=date(2026, 11, 4),
        stories=[
            Story(item(MARKUP, link="javascript:alert(1)", feed=KEYED_FEED), summary=MARKUP),
            Story(item("Data", link="data:text/html,<b>x</b>"), summary="Line one.\n\nLine two."),
            Story(item("Failed"), error=f"HTTP 400 from {MODEL_URL}: {ERROR_BODY}"),
            Story(item("Quiet")),
        ],
        failures=[
            FeedFailure(KEYED_FEED, ERROR_BODY),
            FeedFailure(f"file://{SECRET_PATH}", "not a feed"),
        ],
        usage={
            "steps": [
                {
                    "step": "summarize-1",
                    "tier": "fast",
                    "model": MODEL_URL,
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "cost": None,
                    "ok": False,
                    "error": ERROR_BODY,
                },
                {
                    "step": "summarize-2",
                    "tier": "fast",
                    "model": "gpt-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cost": 0.000123,
                    "ok": True,
                    "error": None,
                },
            ],
            "total_cost": 0.000123,
        },
    )


def test_markdown_renders_feed_and_model_text_as_text_and_links_only_the_web():
    markdown = render_markdown(hostile_issue(), demo=True)

    assert "<script" not in markdown and "<b>" not in markdown
    assert "javascript:" not in markdown and "data:text" not in markdown
    assert "[Failed](<https://example.com/a>)" in markdown
    assert "[Example News](<https://example.com/>)" in markdown
    assert re.search(r"^# \S", markdown, re.M), "one title line"
    assert "Line one." in markdown and "Line two." in markdown
    assert "Summary failed" in markdown and "no text" in markdown
    for secret in SECRETS:
        assert secret not in markdown, secret
    assert "feeds.example.net: 1 problem" in markdown
    assert "example.com: 1 problem" in markdown, "the failed summary"
    assert "a local file: 1 problem" in markdown
    assert "2 model calls, 1 failed; total cost $0.000123, plus a cost unknown for 1 of 2" in (
        markdown
    )


def test_email_escapes_feed_and_model_text_and_shows_no_secret():
    email = render_email(hostile_issue(), demo=True)

    parser = Tags()
    parser.feed(email)
    assert "script" not in parser.tags and "b" not in parser.tags
    assert all(href.startswith(("https://", "http://")) for href in parser.hrefs), parser.hrefs
    for secret in SECRETS:
        assert secret not in email, secret
    assert "feeds.example.net: 1 problem" in email


def test_without_demo_there_is_no_label_and_no_flag():
    issue = hostile_issue()
    for text in (render_markdown(issue), render_email(issue)):
        assert "Automated demo" not in text
        assert UNCHECKED not in text


def test_an_issue_with_no_stories_says_so():
    issue = hostile_issue()
    issue.stories, issue.failures, issue.usage = [], [], {}
    for text in (render_markdown(issue, demo=True), render_email(issue, demo=True)):
        assert "No stories" in text and "No model calls" in text
        assert "Every feed was read" in text


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.hrefs, self.sources = [], [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.hrefs += [value for name, value in attrs if name == "href"]
        self.sources += [value for name, value in attrs if name in ("src", "srcset")]
