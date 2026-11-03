import json
import re
from datetime import UTC, date, datetime
from html.parser import HTMLParser

import httpx
import pytest

from newsletter.editor import (
    CheckedClaim,
    CheckedStory,
    Claim,
    DroppedClaim,
    Edition,
    Prose,
    Theme,
)
from newsletter.llm import LLMClient, LLMError, Settings
from newsletter.render import (
    Newsletter,
    RunReport,
    build_report,
    render_html,
    render_markdown,
    safe_url,
)
from newsletter.sources import FeedFailure, Item

MARKUP_FEED = '<script>alert("feed")</script> Lab <b>News</b>'


def item(link, title, feed_title="Lab News", rights=None):
    return Item(
        title=title,
        link=link,
        summary="text",
        published=datetime(2026, 11, 2, 9, 0, tzinfo=UTC),
        feed_url="https://feeds.example/rss.xml",
        feed_title=feed_title,
        rights=rights,
    )


def checked(claim_id, text, url, status="supported", reason="the passage says so"):
    return CheckedClaim(Claim(claim_id, text, f"passage of {claim_id}", url), status, reason)


S1_LINK = "https://lab.example/model"
S1_OTHER = "https://press.example/lab-model"
S2_LINK = "https://law.example/license"
S3_LINK = "javascript:alert(1)"

S1_TRUE = checked("s1.c1", "The lab released a 3B parameter model.", S1_LINK)
S1_FALSE = checked("s1.c2", "The model scores 95 on the benchmark.", S1_OTHER, "unsupported")
S2_TRUE = checked("s2.c1", "The model is licensed under Apache 2.0.", S2_LINK)
S2_UNCHECKED = checked("s2.c2", "The license changed last week.", S2_LINK, "unchecked", "HTTP 500")
S3_FLAGGED = checked("s3.c1", "A rival will ship <b>next</b> month.", S3_LINK, "unsupported")
DROPPED = DroppedClaim(
    "s1.c3", "s1", "An invented quote.", "not in the source", "passage not found verbatim"
)

STORIES = {
    "s1": [item(S1_LINK, "Lab releases a model", MARKUP_FEED), item(S1_OTHER, "Press on it")],
    "s2": [item(S2_LINK, "The license", "Law Weekly", rights="CC BY 4.0")],
    "s3": [item(S3_LINK, "A rumour", "Rumours")],
    "s4": [item("https://gone.example/a", "A story that failed", "Gone")],
}


def edition():
    s1 = CheckedStory("s1", [S1_TRUE, S1_FALSE], [DROPPED])
    s2 = CheckedStory("s2", [S2_TRUE, S2_UNCHECKED])
    s3 = CheckedStory("s3", [S3_FLAGGED])
    s4 = CheckedStory("s4", error="claims-s4: HTTP 500")
    return Edition(
        [s2, s1, s3, s4],
        lead="s2",
        themes=[Theme("Open models", ["s1.c1"]), Theme("Licensing", ["s2.c1"])],
        intro=Prose("A model and its <i>license</i>.", ["s1.c1", "s2.c1"]),
        why_it_matters={
            "s1": Prose("Small models run locally.", ["s1.c1"]),
            "s2": Prose("You can ship it.", ["s2.c1"]),
        },
    )


def fake_llm():
    """A real client over a fake endpoint: one priced step, one unpriced, one failed."""
    replies = iter(
        [
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0123},
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 5},
                },
            ),
            httpx.Response(400, text="bad request"),
        ]
    )
    settings = Settings(
        base_url="https://llm.example/v1",
        models={"fast": "fast-model-1", "reasoning": "think-model-2"},
        max_retries=0,
    )
    llm = LLMClient(
        settings, retry_backoff=0, transport=httpx.MockTransport(lambda r: next(replies))
    )
    message = [{"role": "user", "content": "x"}]
    llm.complete("claims-s1", message, tier="fast")
    llm.complete("check-s1.c1", message, tier="fast")
    with pytest.raises(LLMError):
        llm.complete("edit", message, tier="reasoning")
    return llm


FAILURES = [FeedFailure("https://broken.example/feed.xml", "HTTP 503")]


def newsletter(the_edition=None, failures=FAILURES):
    the_edition = the_edition or edition()
    report = build_report(failures, the_edition, STORIES, fake_llm().report())
    return Newsletter("Weekly <AI>", date(2026, 11, 3), the_edition, STORIES, report)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.hrefs += [value for name, value in attrs if name == "href"]


def links_of(html):
    parser = Links()
    parser.feed(html)
    parser.close()
    return parser


def source_links():
    return {i.link for items in STORIES.values() for i in items}


# HTML renders, and every link is a story source


def test_html_renders_as_a_responsive_email_and_every_link_is_a_story_source():
    html = render_html(newsletter())
    assert html.startswith("<!DOCTYPE html>")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "@media" in html and "max-width" in html
    parsed = links_of(html)
    assert parsed.hrefs, "the issue has links"
    assert set(parsed.hrefs) <= source_links()
    assert "script" not in parsed.tags


def test_markdown_links_are_all_story_sources():
    markdown = render_markdown(newsletter())
    hrefs = re.findall(r"\]\(<([^>]*)>\)", markdown)
    assert hrefs
    assert set(hrefs) <= source_links()
    # No link outside the `[text](<url>)` form slips past the check above.
    assert markdown.count("](") == len(hrefs)


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_every_shown_claim_links_to_its_source(render):
    out = render(newsletter())
    for claim in (S1_TRUE, S1_FALSE, S2_TRUE, S2_UNCHECKED):
        line = next(line for line in out.splitlines() if claim.claim.text[:20] in line)
        assert claim.claim.source_url in line, claim.id


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_prose_cites_its_claims_with_links_to_their_sources(render):
    out = render(newsletter())
    intro = next(line for line in out.splitlines() if "A model and its" in line)
    assert S1_LINK in intro and S2_LINK in intro
    why = next(line for line in out.splitlines() if "You can ship it." in line)
    assert S2_LINK in why


def test_the_lead_story_comes_first_and_is_marked():
    markdown = render_markdown(newsletter())
    lead = markdown.index("The license")
    assert lead < markdown.index("Lab releases a model") < markdown.index("A rumour")
    assert "Lead story" in markdown[lead - 200 : lead + 200]


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_themes_are_listed(render):
    out = render(newsletter())
    assert "Open models" in out and "Licensing" in out


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_every_source_of_a_story_is_linked_and_its_feed_credited(render):
    out = render(newsletter())
    assert S1_OTHER in out
    assert "CC BY 4.0" in out


# Escaping and link schemes


def test_a_feed_title_with_markup_is_escaped_in_html():
    html = render_html(newsletter())
    assert "<script>" not in html
    assert "&lt;script&gt;alert(&#34;feed&#34;)&lt;/script&gt; Lab &lt;b&gt;News&lt;/b&gt;" in html
    assert "<b>" not in html and "<i>license" not in html
    assert "Weekly &lt;AI&gt;" in html


def test_a_feed_title_with_markup_is_escaped_in_markdown():
    markdown = render_markdown(newsletter())
    assert "<script>" not in markdown and "<b>" not in markdown
    assert r"\<script\>" in markdown


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_a_javascript_link_is_not_rendered(render):
    out = render(newsletter())
    assert "javascript" not in out.lower()
    # The story and its flagged claim are still shown, just without the link.
    assert "A rumour" in out
    assert "A rival will ship" in out


def test_a_quote_in_a_safe_url_cannot_break_out_of_the_attribute():
    url = 'https://lab.example/a"onmouseover="alert(1)'
    story = CheckedStory("s1", [checked("s1.c1", "A claim.", url)])
    stories = {"s1": [item(url, "Title")]}
    report = build_report([], Edition([story]), stories, {})
    html = render_html(Newsletter("T", date(2026, 11, 3), Edition([story]), stories, report))
    assert 'onmouseover="' not in html
    assert links_of(html).hrefs == [url, url]


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        " javascript:alert(1)",
        "java\tscript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
        "mailto:a@example.org",
        "ftp://files.example/a",
        "//evil.example/a",
        "/relative/path",
        "https://",
        "https://ok.example/a b",
        "https://ok.example/a\nb",
        "http://[::1",
        "",
    ],
)
def test_safe_url_refuses_anything_but_an_absolute_http_link(url):
    assert safe_url(url) is None


@pytest.mark.parametrize(
    "url", ["http://ok.example", "https://ok.example/a?b=1&c=2#d", "HTTPS://OK.example/x"]
)
def test_safe_url_keeps_http_and_https_links(url):
    assert safe_url(url) == url


# Flagged claims


def test_a_flagged_claim_is_marked_in_markdown():
    markdown = render_markdown(newsletter())
    lines = markdown.splitlines()

    def line_of(claim):
        # The text before any markup, which the Markdown escapes.
        start = claim.claim.text.split("<")[0]
        return next(line for line in lines if start in line)

    assert "Flagged: unsupported" in line_of(S1_FALSE)
    assert "Flagged: unchecked" in line_of(S2_UNCHECKED)
    assert "Flagged: unsupported" in line_of(S3_FLAGGED)
    assert "Flagged" not in line_of(S1_TRUE)


def test_a_flagged_claim_is_marked_in_html():
    html = render_html(newsletter())
    lines = html.splitlines()

    def line_of(claim):
        return next(line for line in lines if claim.claim.text[:20] in line)

    assert "Flagged: unsupported" in line_of(S1_FALSE)
    assert "Flagged: unchecked" in line_of(S2_UNCHECKED)
    assert "Flagged" not in line_of(S1_TRUE)


# The run report


def test_the_report_lists_a_failed_feed_a_dropped_claim_and_cost_per_step():
    report = newsletter().report
    assert report.failed_sources == FAILURES
    assert [d.id for d in report.dropped_claims] == ["s1.c3"]
    assert [(s["step"], s["model"]) for s in report.steps] == [
        ("claims-s1", "fast-model-1"),
        ("check-s1.c1", "fast-model-1"),
        ("edit", "think-model-2"),
    ]
    assert [s["cost"] for s in report.steps] == [0.0123, None, None]
    assert report.total_cost == pytest.approx(0.0123)
    assert report.cost_complete is False
    assert report.failed_steps == ["edit"]


def test_the_report_lists_flagged_claims_and_dropped_stories():
    report = newsletter().report
    assert [(f.id, f.status) for f in report.flagged_claims] == [
        ("s2.c2", "unchecked"),
        ("s1.c2", "unsupported"),
        ("s3.c1", "unsupported"),
    ]
    assert [(d.story_id, d.title, d.reason) for d in report.dropped_stories] == [
        ("s4", "A story that failed", "claims-s4: HTTP 500")
    ]
    assert report.ok is False


def test_a_story_whose_claims_were_all_dropped_is_a_dropped_story():
    story = CheckedStory("s1", [], [DROPPED])
    report = build_report([], Edition([story]), STORIES, {})
    assert [(d.story_id, d.reason) for d in report.dropped_stories] == [
        ("s1", "every claim was dropped (1)")
    ]


def test_a_story_with_no_text_is_a_dropped_story():
    report = build_report([], Edition([CheckedStory("s1")]), STORIES, {})
    assert report.dropped_stories[0].reason == "no claims: the sources gave no text"


def test_a_clean_run_is_ok():
    story = CheckedStory("s2", [S2_TRUE])
    report = build_report([], Edition([story]), STORIES, {})
    assert report.ok is True
    assert report.steps == [] and report.total_cost == 0 and report.cost_complete is True


def test_the_report_carries_the_editor_error_and_rejections():
    failed = Edition([CheckedStory("s2", [S2_TRUE])], error="edit: HTTP 500")
    failed.rejections.append("edit: unknown claim id 'x'")
    report = build_report([], failed, STORIES, {})
    assert report.editor_error == "edit: HTTP 500"
    assert report.editor_rejections == ["edit: unknown claim id 'x'"]
    assert report.ok is False


def test_the_report_is_plain_data():
    data = newsletter().report.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["failed_sources"] == [{"url": FAILURES[0].url, "reason": "HTTP 503"}]


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_the_rendered_report_shows_failures_and_cost_per_step(render):
    out = render(newsletter())
    report = out[out.index("Run report") :]
    for text in (
        "https://broken.example/feed.xml",
        "HTTP 503",
        "s1.c3",
        "passage not found verbatim",
        "s2.c2",
        "HTTP 500",
        "A story that failed",
        "claims-s1",
        "check-s1.c1",
        "fast-model-1",
        "think-model-2",
        "$0.012300",
        "unknown",
        "failed: HTTP 400",
        "Total cost: $0.012300, plus a cost unknown for 2 of 3 steps",
    ):
        assert text in report, text


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_a_failed_editor_pass_is_said_and_the_stories_still_shown(render):
    failed = Edition([CheckedStory("s1", [S1_TRUE]), CheckedStory("s2", [S2_TRUE])])
    failed.error = "edit: HTTP 500"
    out = render(newsletter(failed))
    assert "editor pass failed" in out
    assert out.index("Lab releases a model") < out.index("The license")
    assert S1_TRUE.claim.text in out and S2_TRUE.claim.text in out


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_an_issue_with_nothing_to_show_says_so(render):
    empty = Edition([])
    report = build_report(FAILURES, empty, {}, {})
    out = render(Newsletter("T", date(2026, 11, 3), empty, {}, report))
    assert "No stories this time" in out
    assert "No model calls were made." in out
    assert "HTTP 503" in out


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_a_status_with_no_label_is_still_flagged(render):
    odd = CheckedClaim(Claim("s2.c1", "An odd claim.", "p", S2_LINK), "disputed", "r")
    story = Edition([CheckedStory("s2", [odd])])
    out = render(Newsletter("T", date(2026, 11, 3), story, STORIES, RunReport()))
    line = next(line for line in out.splitlines() if "An odd claim." in line)
    assert "Flagged: disputed" in line


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_a_story_with_no_source_items_is_shown_by_id_with_no_link(render):
    story = Edition([CheckedStory("s9", [checked("s9.c1", "A claim.", S1_LINK)])])
    out = render(Newsletter("T", date(2026, 11, 3), story, {}, RunReport()))
    assert "s9" in out
    assert out.count(S1_LINK) == 1


def test_markdown_code_spans_survive_backticks():
    failure = FeedFailure("https://x.example/`a``b", "HTTP 404")
    report = build_report([failure], Edition([]), {}, {})
    markdown = render_markdown(Newsletter("T", date(2026, 11, 3), Edition([]), {}, report))
    assert "``` https://x.example/`a``b ```" in markdown


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_rejected_editor_replies_are_in_the_rendered_report(render):
    the_edition = edition()
    the_edition.rejections.append("edit: unknown claim id 'zz'")
    out = render(newsletter(the_edition))
    assert "unknown claim id" in out[out.index("Run report") :]


@pytest.mark.parametrize("render", [render_html, render_markdown])
def test_a_clean_run_says_so(render):
    story = Edition([CheckedStory("s2", [S2_TRUE])])
    report = build_report([], story, STORIES, {})
    out = render(Newsletter("T", date(2026, 11, 3), story, STORIES, report))
    assert "Every feed was read" in out
    assert "Flagged" not in out
