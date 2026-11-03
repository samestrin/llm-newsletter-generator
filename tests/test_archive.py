from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

import newsletter.archive.site as site_module
from newsletter.archive import ArchiveError, build_site, issue_path
from newsletter.issue import Issue, Story
from newsletter.sources import FeedFailure, Item

API_KEY = "sk-test-SECRET-archive-0123456789"
ERROR_BODY = f'{{"error": "bad request, key {API_KEY}"}}'
SECRET_PATH = "/home/runner/.cache/newsletter/feeds/secret.json"
MODEL_URL = "http://model.internal:8080/v1/chat/completions"
MARKUP = '<script>alert("x")</script><b>Bold</b> & co'


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


def fixture_issue(day=date(2026, 11, 3), **overrides):
    values = dict(
        title="AI Daily",
        topic="AI",
        issue_date=day,
        stories=[
            Story(item("First story"), summary="First summary."),
            Story(item("Second story", link="https://lab.example.org/b"), summary="Second."),
        ],
        failures=[],
        usage={"steps": [], "total_cost": 0.0},
    )
    values.update(overrides)
    return Issue(**values)


def seeded_issue():
    """An issue whose feeds, model calls and errors carry an error body, a path and a key."""
    return fixture_issue(
        day=date(2026, 11, 4),
        title=f"{MARKUP} ../../../etc/passwd",
        stories=[
            Story(
                item(
                    title=MARKUP,
                    link="javascript:alert(1)",
                    feed=f"https://user:{API_KEY}@feeds.example.net/rss?key={API_KEY}",
                ),
                error=f"summarize-1: HTTP 400 from {MODEL_URL}: {ERROR_BODY}",
            ),
            Story(item("../../escape", link="../../../outside.html"), summary=MARKUP),
            Story(item("Data link", link="data:text/html,<b>x</b>"), summary="ok"),
        ],
        failures=[
            FeedFailure(f"https://feeds.example.net/rss?key={API_KEY}", ERROR_BODY),
            FeedFailure("https://feeds.example.net/other", f"cannot read {SECRET_PATH}"),
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


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.hrefs += [value for name, value in attrs if name == "href"]


def parse(path: Path) -> Links:
    parser = Links()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def pages(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_builds_index_and_one_page_per_issue(tmp_path):
    out = tmp_path / "site"
    issues = [fixture_issue(date(2026, 11, 1)), fixture_issue(date(2026, 11, 3))]

    build_site(issues, out)

    assert [p.relative_to(out).as_posix() for p in pages(out)] == [
        "index.html",
        "issues/2026-11-01/index.html",
        "issues/2026-11-03/index.html",
    ]
    index = (out / "index.html").read_text(encoding="utf-8")
    assert index.index("2026-11-03") < index.index("2026-11-01"), "newest first"
    assert parse(out / "index.html").hrefs == [
        "issues/2026-11-03/index.html",
        "issues/2026-11-01/index.html",
    ]
    page = (out / "issues/2026-11-03/index.html").read_text(encoding="utf-8")
    assert "First story" in page and "First summary." in page and "Second." in page
    assert "https://example.com/a" in parse(out / "issues/2026-11-03/index.html").hrefs
    assert "Every feed was read" in page


def test_page_path_comes_from_the_date_only():
    assert issue_path(fixture_issue(title="../../x")) == "issues/2026-11-03/index.html"


def test_two_issues_on_one_date_fail_the_build(tmp_path):
    with pytest.raises(ArchiveError, match="2026-11-03"):
        build_site([fixture_issue(), fixture_issue()], tmp_path / "site")
    assert not (tmp_path / "site").exists()


def test_markup_and_dot_dot_slug_render_as_text_inside_the_site(tmp_path):
    out = tmp_path / "site"
    build_site([seeded_issue()], out)

    files = pages(out)
    assert all(p.resolve().is_relative_to(out.resolve()) for p in files)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["site"], "nothing written beside it"
    page = out / "issues/2026-11-04/index.html"
    text = page.read_text(encoding="utf-8")
    assert "<script" not in text and "<b>" not in text
    assert (
        "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;&lt;b&gt;Bold&lt;/b&gt; &amp; co" in text
    )
    assert "../../escape" in text
    assert "script" not in parse(page).tags and "b" not in parse(page).tags


def test_only_http_and_https_links_are_rendered(tmp_path):
    out = tmp_path / "site"
    build_site([seeded_issue()], out)

    for path in pages(out):
        for href in parse(path).hrefs:
            assert href.startswith(("https://", "http://", "issues/", "../../")), href
    text = (out / "issues/2026-11-04/index.html").read_text(encoding="utf-8")
    assert "javascript:" not in text and "data:text" not in text
    assert "outside.html" not in text


def test_secrets_do_not_appear_in_any_page(tmp_path):
    out = tmp_path / "site"
    build_site([seeded_issue(), fixture_issue()], out)

    for path in pages(out):
        text = path.read_text(encoding="utf-8")
        for secret in (
            API_KEY,
            "SECRET",
            "bad request",
            SECRET_PATH,
            "secret.json",
            "model.internal",
        ):
            assert secret not in text, (path, secret)


def test_run_report_counts_failures_by_feed_host(tmp_path):
    out = tmp_path / "site"
    build_site([seeded_issue()], out)

    text = (out / "issues/2026-11-04/index.html").read_text(encoding="utf-8")
    assert "feeds.example.net: 3 problems" in text  # two feed failures and one failed summary
    assert "a local file: 1 problem" in text
    assert "2 model calls, 1 failed" in text
    assert "$0.000123" in text


def test_a_failed_build_leaves_the_previous_site_untouched(tmp_path, monkeypatch):
    out = tmp_path / "site"
    build_site([fixture_issue()], out)
    before = {p: p.read_bytes() for p in pages(out)}

    real = site_module.render_issue

    def boom(issue):
        if issue.issue_date == date(2026, 11, 4):
            raise RuntimeError("disk full")
        return real(issue)

    monkeypatch.setattr(site_module, "render_issue", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        build_site([fixture_issue(), seeded_issue()], out)

    assert {p: p.read_bytes() for p in pages(out)} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["site"], "temp dir removed"


def test_a_rebuild_replaces_the_previous_site(tmp_path):
    out = tmp_path / "site"
    build_site([fixture_issue(date(2026, 11, 1))], out)
    build_site([fixture_issue(date(2026, 11, 2))], out)

    assert [p.relative_to(out).as_posix() for p in pages(out)] == [
        "index.html",
        "issues/2026-11-02/index.html",
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["site"]


def test_an_issue_with_no_stories_still_gets_a_page(tmp_path):
    out = tmp_path / "site"
    build_site([fixture_issue(stories=[])], out)

    text = (out / "issues/2026-11-03/index.html").read_text(encoding="utf-8")
    assert "No stories" in text
    assert "No model calls" in text


def test_an_empty_archive_has_an_index(tmp_path):
    out = tmp_path / "site"
    build_site([], out)

    assert "No issues yet" in (out / "index.html").read_text(encoding="utf-8")


def test_a_malformed_link_is_left_out_without_failing_the_build(tmp_path):
    out = tmp_path / "site"
    bad = [
        Story(item("Bad port", link="https://example.com:99999/x"), summary="ok"),
        Story(item("Bad host", link="http://[::1/x", feed="https://[::1/feed"), summary="ok"),
        Story(item("IPv6", link="http://[::1]:8080/x", feed="http://[::1]:8080/f"), summary="ok"),
    ]
    build_site([fixture_issue(stories=bad)], out)

    hrefs = parse(out / "issues/2026-11-03/index.html").hrefs
    assert "https://example.com:99999/x" not in hrefs
    assert "http://[::1]:8080/x" in hrefs
    assert "http://[::1]:8080/" in hrefs


def test_a_failed_swap_puts_the_previous_site_back(tmp_path, monkeypatch):
    out = tmp_path / "site"
    build_site([fixture_issue()], out)
    before = {p: p.read_bytes() for p in pages(out)}

    real = site_module.os.rename

    def rename(src, dst):
        if Path(src).name.startswith(".site-build-"):
            raise OSError("rename failed")
        real(src, dst)

    monkeypatch.setattr(site_module.os, "rename", rename)
    with pytest.raises(OSError, match="rename failed"):
        build_site([fixture_issue(date(2026, 11, 5))], out)

    assert {p: p.read_bytes() for p in pages(out)} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["site"]


def test_a_story_with_no_text_says_so(tmp_path):
    out = tmp_path / "site"
    build_site([fixture_issue(stories=[Story(item("Quiet"))])], out)

    assert "no text" in (out / "issues/2026-11-03/index.html").read_text(encoding="utf-8")
