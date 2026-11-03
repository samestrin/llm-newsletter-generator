import json
import re
from datetime import date
from pathlib import Path

import httpx
import pytest

from newsletter.issue import fill, load_prompt, make_issue, render_markdown, summarize
from newsletter.llm import LLMClient, Settings
from newsletter.sources import FeedFailure, Item, RssSource
from newsletter.sources.http import FetchError, HttpResponse

FIXTURES = Path(__file__).parent / "fixtures" / "feeds"
RSS = "https://example.com/feed.xml"
ATOM = "https://lab.example.org/atom.xml"
API_KEY = "sk-test-SECRET-issue-0123456789"
LINK = re.compile(r"\[[^\]]*\]\(<([^>]*)>\)")


def fake_web(routes):
    def fetch(url, headers, timeout):
        answer = routes.get(url, HttpResponse(404))
        if isinstance(answer, Exception):
            raise answer
        return answer

    return fetch


class FakeModel:
    """An OpenAI-compatible endpoint that summarizes by echoing the story title.

    A story whose title contains one of `fail_on` gets an HTTP 400.
    """

    def __init__(self, fail_on=()):
        self.fail_on = fail_on
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        prompt = body["messages"][-1]["content"]
        title = re.search(r"title `([^`]*)`", prompt).group(1)
        if any(word in title for word in self.fail_on):
            return httpx.Response(400, json={"error": f"bad request, key {API_KEY}"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"  Summary of {title}.  "}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )


def make_llm(model, **env):
    settings = Settings.from_env(
        {
            "NEWSLETTER_BASE_URL": "http://model.test/v1",
            "NEWSLETTER_API_KEY": API_KEY,
            "NEWSLETTER_FAST_MODEL": "fast-1",
            "NEWSLETTER_FAST_PRICE_INPUT": "1",
            "NEWSLETTER_FAST_PRICE_OUTPUT": "2",
            "NEWSLETTER_MAX_RETRIES": "0",
            **env,
        }
    )
    return LLMClient(settings, transport=httpx.MockTransport(model))


def fixture_source():
    return RssSource(
        fetch=fake_web(
            {
                RSS: HttpResponse(200, {}, (FIXTURES / "rss.xml").read_bytes()),
                ATOM: HttpResponse(200, {}, (FIXTURES / "atom.xml").read_bytes()),
                "https://slow.example/feed": FetchError("timed out after 10s"),
            }
        )
    )


def item(title="A story", link="https://example.com/a", summary="Some text.", **extra):
    fields = {
        "published": None,
        "feed_url": RSS,
        "feed_title": "Example AI News",
        "rights": None,
        **extra,
    }
    return Item(title=title, link=link, summary=summary, **fields)


def test_an_issue_from_fixture_feeds_links_every_item():
    model = FakeModel()
    issue = make_issue(
        [RSS, ATOM],
        title="AI Weekly",
        topic="AI",
        source=fixture_source(),
        llm=make_llm(model),
        issue_date=date(2026, 11, 3),
    )
    markdown = render_markdown(issue)

    assert markdown.startswith("# AI Weekly\n")
    assert "2026-11-03" in markdown
    stories = markdown.split("## Sources", 1)[0]
    headings = [line for line in stories.splitlines() if line.startswith("## ")]
    assert all(LINK.fullmatch(h.removeprefix("## ")) for h in headings)
    assert LINK.findall(stories) == [
        "https://example.com/posts/open-model",
        "https://example.com/posts/benchmarks",
        "https://lab.example.org/scaling",
    ]
    for story in issue.stories:
        assert f"Summary of {story.item.title}." in markdown
    # One fast-model call per item, with the item's text and the topic in the prompt.
    assert len(model.requests) == 3
    assert {r["model"] for r in model.requests} == {"fast-1"}
    first = model.requests[0]["messages"][-1]["content"]
    assert "`AI`" in first and "The model is open & small." in first


def test_failed_feeds_and_dropped_entries_are_listed_in_the_issue():
    issue = make_issue(
        [RSS, "https://slow.example/feed"],
        title="AI Weekly",
        topic="AI",
        source=fixture_source(),
        llm=make_llm(FakeModel()),
        issue_date=date(2026, 11, 3),
    )
    markdown = render_markdown(issue)

    report = markdown.split("## Run report", 1)[1]
    assert "https://slow.example/feed" in report and "timed out after 10s" in report
    assert "An item with no link" in report
    assert not issue.ok


def test_a_failed_model_call_keeps_the_item_and_is_reported():
    issue = make_issue(
        [RSS, ATOM],
        title="AI Weekly",
        topic="AI",
        source=fixture_source(),
        llm=make_llm(FakeModel(fail_on=("Benchmarks",))),
        issue_date=date(2026, 11, 3),
    )
    markdown = render_markdown(issue)

    # Still linked, never skipped.
    assert "https://example.com/posts/benchmarks" in LINK.findall(markdown)
    failed = [s for s in issue.stories if s.error]
    assert [s.item.title for s in failed] == ["Benchmarks revisited"]
    body, report = markdown.split("## Run report", 1)
    assert "summary failed" in body.lower()
    assert "summarize-2" in report and "HTTP 400" in report
    assert API_KEY not in markdown
    assert not issue.ok


def test_run_report_lists_models_tokens_and_cost():
    issue = make_issue(
        [ATOM],
        title="AI Weekly",
        topic="AI",
        source=fixture_source(),
        llm=make_llm(FakeModel()),
        issue_date=date(2026, 11, 3),
    )
    report = render_markdown(issue).split("## Run report", 1)[1]

    assert "| summarize-1 | fast | fast-1 | 100 | 20 | $0.000140 | ok |" in report
    assert "Total cost: $0.000140" in report
    assert issue.ok


def test_unknown_cost_is_not_reported_as_a_total():
    issue = make_issue(
        [ATOM],
        title="AI Weekly",
        topic="AI",
        source=fixture_source(),
        llm=make_llm(FakeModel(), NEWSLETTER_FAST_PRICE_INPUT="", NEWSLETTER_FAST_PRICE_OUTPUT=""),
        issue_date=date(2026, 11, 3),
    )
    report = render_markdown(issue).split("## Run report", 1)[1]

    assert "| unknown |" in report
    assert "unknown for 1 of 1 steps" in report


def test_feed_rights_are_credited():
    issue = make_issue(
        [RSS, ATOM],
        title="AI Weekly",
        topic="AI",
        source=fixture_source(),
        llm=make_llm(FakeModel()),
        issue_date=date(2026, 11, 3),
    )
    markdown = render_markdown(issue)

    sources = markdown.split("## Sources", 1)[1].split("## Run report", 1)[0]
    assert LINK.findall(sources) == [RSS, ATOM]
    assert "Example AI News" in sources and "Example Lab Blog" in sources
    assert "Copyright 2026 Example Media. Headlines and links only." in markdown
    assert "CC BY 4.0" in markdown


def test_no_items_is_not_ok_and_says_so():
    issue = make_issue(
        ["https://slow.example/feed"],
        title="AI Weekly",
        topic="AI",
        source=fixture_source(),
        llm=make_llm(FakeModel()),
        issue_date=date(2026, 11, 3),
    )

    assert not issue.ok
    assert "No stories" in render_markdown(issue)


def test_an_item_without_text_is_linked_without_asking_the_model():
    model = FakeModel()
    stories = summarize([item(summary="")], make_llm(model), topic="AI")

    assert model.requests == []
    assert stories[0].summary is None and stories[0].error is None


def test_long_item_text_is_cut_before_it_reaches_the_model():
    model = FakeModel()
    summarize([item(summary="word " * 10_000)], make_llm(model), topic="AI")

    assert len(model.requests[0]["messages"][-1]["content"]) < 10_000


def test_markdown_in_titles_and_links_cannot_break_the_link():
    tricky = item(
        title="Model [beta] *launch*\nsecond line",
        link="https://example.com/a (b)<c>",
        summary="",
    )
    issue = make_issue(
        [],
        title="AI Weekly",
        topic="AI",
        source=RssSource(fetch=fake_web({})),
        llm=make_llm(FakeModel()),
        issue_date=date(2026, 11, 3),
    )
    issue.stories = summarize([tricky], make_llm(FakeModel()), topic="AI")
    markdown = render_markdown(issue)

    assert r"[Model \[beta\] \*launch\* second line](<https://example.com/a%20(b)%3Cc%3E>)" in (
        markdown
    )


def test_feed_failure_reasons_are_shown_as_code():
    issue = make_issue(
        [],
        title="AI Weekly",
        topic="AI",
        source=RssSource(fetch=fake_web({})),
        llm=make_llm(FakeModel()),
        issue_date=date(2026, 11, 3),
    )
    issue.failures = [FeedFailure("https://x.example/`feed`", "HTTP 404")]

    assert "- `` https://x.example/`feed` ``: HTTP 404" in render_markdown(issue)


def test_fill_replaces_each_placeholder_once():
    template = "{{ a }} and {{b}} and {{ a }}"
    assert fill(template, a="{{ b }}", b="2") == "{{ b }} and 2 and {{ b }}"


def test_fill_rejects_a_missing_value():
    with pytest.raises(KeyError, match="topic"):
        fill("about {{ topic }}")


def test_the_item_prompt_ships_with_the_package():
    prompt = load_prompt("item")
    assert {"topic", "item_title", "item_description"} <= set(
        re.findall(r"\{\{\s*(\w+)\s*\}\}", prompt)
    )
