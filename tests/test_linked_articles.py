"""Link-only items (a Hacker News story, an hnrss entry, an RSS teaser of a line) get the
linked article's main text as their passage; an article that cannot be read is skipped."""

import json
from datetime import date
from pathlib import Path

import pytest

from newsletter.editor import Edition
from newsletter.render import Newsletter, build_report, render_html, render_markdown
from newsletter.sources import HackerNewsSource, Item, LinkedArticle, PageSource, read_linked
from newsletter.sources.feed import parse_feed
from newsletter.sources.http import HttpResponse

PAGES = Path(__file__).parent / "fixtures" / "pages"
API = "https://hn.example/v0"
ARTICLE = "https://a.example/open-model"
HNRSS = "https://hnrss.org/newest?points=300&count=5"

HNRSS_FEED = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Hacker News: Newest</title><link>https://news.ycombinator.com/</link>
<item>
  <title>Open model ships</title>
  <link>{ARTICLE}</link>
  <description><![CDATA[
<p>Article URL: <a href="{ARTICLE}">{ARTICLE}</a></p>
<p>Comments URL: <a href="https://news.ycombinator.com/item?id=1">https://news.ycombinator.com/item?id=1</a></p>
<p>Points: 710</p>
<p># Comments: 312</p>
]]></description>
</item>
</channel></rss>""".encode()


def html(name: str) -> HttpResponse:
    return HttpResponse(200, {"content-type": "text/html"}, (PAGES / name).read_bytes())


class FakeWeb:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, url, headers, timeout):
        self.requests.append(url)
        answer = self.routes.get(url, HttpResponse(404))
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr("newsletter.sources.http.resolve_host", lambda host: ["93.184.215.14"])


def item(summary: str, link: str = ARTICLE, title: str = "Open model ships") -> Item:
    return Item(title, link, summary, None, "https://feed.example/rss", "Feed", None)


def hn_routes(story) -> dict:
    return {
        f"{API}/topstories.json": HttpResponse(200, {}, b"[1]"),
        f"{API}/item/1.json": HttpResponse(200, {}, json.dumps(story).encode()),
    }


STORY = {"id": 1, "type": "story", "title": "Open model ships", "url": ARTICLE, "score": 710}


def test_a_hacker_news_story_with_a_link_gets_its_page_text_as_the_passage():
    web = FakeWeb({**hn_routes(STORY), ARTICLE: html("article.html")})
    [story] = HackerNewsSource(api_url=API, fetch=web).collect().items
    assert story.summary == "Open model ships"  # The API gives only the title.

    linked = read_linked([story], PageSource(fetch=web))

    [got] = linked.items
    assert got.summary.startswith("Open model ships\nAn open-weights model was released")
    assert "It scores 81.2 on the benchmark" in got.summary
    assert got.title == "Open model ships"
    assert got.link == ARTICLE
    assert got.feed_title == "Hacker News"  # Still credited to the source that listed it.
    assert linked.articles == [LinkedArticle("Open model ships", ARTICLE, "fetched", None)]


def test_a_failed_page_fetch_is_listed_as_skipped_and_never_summarized():
    web = FakeWeb({**hn_routes(STORY), ARTICLE: HttpResponse(503)})
    stories = HackerNewsSource(api_url=API, fetch=web).collect().items

    linked = read_linked(stories, PageSource(fetch=web))

    assert linked.items == []
    assert linked.articles == [LinkedArticle("Open model ships", ARTICLE, "skipped", "HTTP 503")]


def test_a_page_robots_forbids_is_skipped_not_fetched():
    robots = HttpResponse(200, {"content-type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
    web = FakeWeb({"https://a.example/robots.txt": robots, ARTICLE: html("article.html")})

    linked = read_linked([item("Open model ships")], PageSource(fetch=web))

    assert linked.items == []
    [article] = linked.articles
    assert article.status == "skipped" and "robots.txt" in article.reason
    assert ARTICLE not in web.requests


def test_an_hnrss_entry_is_an_aggregator_wrapper_and_gets_the_article():
    feed = parse_feed(HNRSS_FEED, HNRSS)
    web = FakeWeb({ARTICLE: html("article.html")})

    linked = read_linked(feed.items, PageSource(fetch=web))

    [got] = linked.items
    assert "710" not in got.summary and "Comments URL" not in got.summary
    assert "It scores 81.2 on the benchmark" in got.summary
    assert got.feed_url == HNRSS
    assert [a.status for a in linked.articles] == ["fetched"]


def test_an_item_with_its_own_text_is_kept_as_it_is_and_not_fetched():
    text = (
        "The lab released a 3B parameter model on Tuesday. It runs on a laptop, scores 71 on "
        "the benchmark and is licensed under Apache 2.0."
    )
    own = item(f"Open model ships\n{text}")
    web = FakeWeb({})

    linked = read_linked([own], PageSource(fetch=web))

    assert linked.items == [own]
    assert linked.articles == []
    assert web.requests == []


@pytest.mark.parametrize(
    "summary",
    [
        "",
        "Open model ships",
        "Open model ships\nhttps://a.example/open-model",
        "Points: 710\n# Comments: 312",
        "A new model.",  # A line under the threshold is a teaser, not the story.
    ],
)
def test_title_metadata_or_a_line_alone_is_link_only(summary):
    web = FakeWeb({ARTICLE: html("article.html")})
    linked = read_linked([item(summary)], PageSource(fetch=web))
    assert [a.status for a in linked.articles] == ["fetched"]


def test_items_named_in_keep_are_never_fetched():
    """A web page source item is already the article; it is not read twice."""
    page = item("Short page.", link=ARTICLE)
    web = FakeWeb({})
    linked = read_linked([page], PageSource(fetch=web), keep={"https://feed.example/rss"})
    assert linked.items == [page] and linked.articles == [] and web.requests == []


def test_fetched_and_skipped_articles_are_in_the_run_report():
    articles = [
        LinkedArticle("Open model ships", ARTICLE, "fetched", None),
        LinkedArticle("Gone story", "https://gone.example/x", "skipped", "HTTP 503"),
    ]
    report = build_report([], Edition(stories=[]), {}, {}, linked=articles)

    assert report.linked_articles == articles
    assert report.ok is False
    assert report.to_dict()["linked_articles"][1]["reason"] == "HTTP 503"
    newsletter = Newsletter("Issue", date(2026, 11, 19), Edition([]), {}, report)
    markdown = render_markdown(newsletter)
    assert "**Linked articles**" in markdown
    assert "(<https://a.example/open-model>): fetched" in markdown
    assert "(<https://gone.example/x>): skipped, HTTP 503" in markdown
    page = render_html(newsletter)
    assert "Linked articles" in page and "skipped, HTTP 503" in page


def test_only_fetched_articles_keep_the_report_ok():
    fetched = [LinkedArticle("Open model ships", ARTICLE, "fetched", None)]
    assert build_report([], Edition(stories=[]), {}, {}, linked=fetched).ok is True
