"""Feeds checked before the run, and run limits: `POST /api/feeds/check`, the re-check
in `POST /api/runs`, demo mode's limits, a busy server's 409, the check's rate cap and
`GET /api/presets`."""

import datetime as dt
import threading

import pytest
from fastapi.testclient import TestClient

from newsletter.api import ApiSettings, create_app
from newsletter.api import feed_check as fc
from newsletter.api.pipeline import default_runner
from newsletter.issue import gather_stories
from newsletter.progress import no_progress
from newsletter.sources.http import FetchError, HttpResponse, TooLarge
from newsletter.sources.rss import FeedFailure
from test_api import TOKEN, FakeRunner, headers, make_settings

GOOD = "https://good.example.org/feed.xml"
ALSO_GOOD = "https://also.example.org/atom.xml"
EMPTY = "https://empty.example.org/feed.xml"
PAGE = "https://page.example.org/"
DOWN = "https://down.example.org/feed.xml"
SHY = "https://shy.example.org/feed.xml"
MISSING = "https://missing.example.org/feed.xml"
BROKEN = "https://broken.example.org/feed.xml"
HUGE = "https://huge.example.org/feed.xml"
TO_LOOPBACK = "https://hop.example.org/feed.xml"

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Good news</title>
<item><title>One</title><link>https://good.example.org/1</link></item>
<item><title>Two</title><link>https://good.example.org/2</link></item>
</channel></rss>"""
ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Also good</title>
<entry><title>Three</title><link href="https://also.example.org/3"/><id>3</id>
<updated>2026-11-20T00:00:00Z</updated></entry>
</feed>"""
NO_ITEMS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Quiet</title></channel></rss>"""
HTML = b"<!doctype html><html><head><title>Home</title></head><body><p>Hello</p></body></html>"
MALFORMED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Half</title><item><title>One</title>"""


class Web:
    """A fake web: each feed above, robots.txt that turns `SHY` away, and a count of
    every request."""

    def __init__(self):
        self.asked = []

    def __call__(self, url, headers, timeout):
        self.asked.append(url)
        if url == "https://shy.example.org/robots.txt":
            return HttpResponse(200, {}, b"User-agent: *\nDisallow: /\n")
        if url.endswith("/robots.txt"):
            return HttpResponse(404)
        if url.startswith("https://down.example.org/"):
            raise FetchError("timed out after 10s")
        if url == HUGE:
            raise TooLarge("response larger than 10485760 bytes")
        if url == TO_LOOPBACK:
            return HttpResponse(302, {"location": "http://127.0.0.1/feed.xml"})
        bodies = {GOOD: RSS, ALSO_GOOD: ATOM, EMPTY: NO_ITEMS, PAGE: HTML, BROKEN: MALFORMED}
        if url in bodies:
            return HttpResponse(200, {}, bodies[url])
        return HttpResponse(404)


def public(host):
    return {"localhost": ["127.0.0.1"], "intranet.example.org": ["10.0.0.7"]}.get(
        host, ["93.184.216.34"]
    )


def check(url, web=None):
    return fc.check_feed(url, fetch=web or Web(), resolve=public)


# The check itself, against fake feeds.


def test_a_feed_with_items_is_ok_with_its_title_and_item_count():
    assert check(GOOD) == fc.FeedCheck(ok=True, title="Good news", items=2, error=None)
    assert check(ALSO_GOOD) == fc.FeedCheck(ok=True, title="Also good", items=1, error=None)


@pytest.mark.parametrize(
    "url,sentence",
    [
        (DOWN, "This address did not answer"),
        (PAGE, "This is a web page, not a feed"),
        (EMPTY, "This feed has no items"),
        (SHY, "The site asks not to be read"),
    ],
)
def test_each_of_the_four_error_cases_returns_its_sentence(url, sentence):
    result = check(url)
    assert result == fc.FeedCheck(ok=False, title=None, items=0, error=sentence)


@pytest.mark.parametrize(
    "url,sentence",
    [
        (MISSING, "This address answered with an error (HTTP 404)"),
        (BROKEN, "This feed is broken and cannot be read"),
        (HUGE, "This feed is larger than 10 MB"),
        ("ftp://good.example.org/feed.xml", "Only http and https addresses can be read"),
        ("not a url", "Only http and https addresses can be read"),
        ("https://unknown.example.org/feed.xml", "This address did not answer"),
    ],
)
def test_every_other_failure_is_a_plain_sentence_too(url, sentence):
    def resolve(host):
        if host == "unknown.example.org":
            raise FetchError("cannot resolve unknown.example.org")
        return public(host)

    result = fc.check_feed(url, fetch=Web(), resolve=resolve)
    assert result == fc.FeedCheck(ok=False, title=None, items=0, error=sentence)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/feed.xml",
        "http://localhost:8000/feed.xml",
        "http://[::1]/feed.xml",
        "https://intranet.example.org/feed.xml",
        TO_LOOPBACK,
    ],
)
def test_a_loopback_or_private_address_is_refused_without_a_crash(url):
    web = Web()
    result = fc.check_feed(url, fetch=web, resolve=public)
    assert result.ok is False
    assert result.error == fc.PRIVATE
    assert not any("127.0.0.1" in asked or "10.0.0.7" in asked for asked in web.asked)


def test_the_check_reads_robots_txt_before_the_feed_and_never_fetches_a_shy_one():
    web = Web()
    fc.check_feed(SHY, fetch=web, resolve=public)
    assert web.asked == ["https://shy.example.org/robots.txt"]


def test_the_check_sends_the_feeds_user_agent():
    sent = []

    def web(url, headers, timeout):
        sent.append(headers.get("User-Agent"))
        return Web()(url, headers, timeout)

    fc.check_feed(GOOD, fetch=web, resolve=public)
    assert sent and all(agent and agent.startswith("newsletter/") for agent in sent)


# The check over HTTP.


@pytest.fixture
def web():
    return Web()


@pytest.fixture
def clock():
    return Clock()


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def api(tmp_path, web, clock=None, runner=None, **env) -> TestClient:
    settings = make_settings(tmp_path, **env)
    app = create_app(
        settings,
        runner=runner or FakeRunner(),
        fetch=web,
        resolve=public,
        **({"clock": clock} if clock else {}),
    )
    return TestClient(app)


def test_the_check_answers_ok_title_items_and_error(tmp_path, web):
    with api(tmp_path, web) as client:
        good = client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
        empty = client.post("/api/feeds/check", json={"url": EMPTY}, headers=headers())
    assert good.status_code == 200
    assert good.json() == {"ok": True, "title": "Good news", "items": 2, "error": None}
    assert empty.json() == {"ok": False, "title": None, "items": 0, "error": fc.NO_ITEMS}


def test_a_loopback_url_is_refused_over_http_too(tmp_path, web):
    with api(tmp_path, web) as client:
        response = client.post(
            "/api/feeds/check", json={"url": "http://127.0.0.1:8000/api/health"}, headers=headers()
        )
    assert response.status_code == 200
    assert response.json()["error"] == fc.PRIVATE
    assert web.asked == []


def test_the_check_needs_the_token_and_a_url(tmp_path, web):
    with api(tmp_path, web) as client:
        no_token = client.post(
            "/api/feeds/check", json={"url": GOOD}, headers={"Origin": headers()["Origin"]}
        )
        no_url = client.post("/api/feeds/check", json={}, headers=headers())
        extra = client.post("/api/feeds/check", json={"url": GOOD, "x": 1}, headers=headers())
    assert no_token.status_code == 401
    assert no_url.status_code == 422
    assert extra.status_code == 422
    assert web.asked == []


def test_the_check_makes_no_model_call_and_needs_no_model_key(tmp_path, web):
    with api(tmp_path, web, NEWSLETTER_API_KEY="") as client:
        response = client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
    assert response.json()["ok"] is True
    assert all("example.org" in url for url in web.asked)


def test_calls_under_the_cap_pass_and_calls_past_it_are_429(tmp_path, web, clock):
    with api(tmp_path, web, clock) as client:
        answers = [
            client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
            for _ in range(30)
        ]
        clock.now += 20
        over = client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
        asked = len(web.asked)
        clock.now += 40
        again = client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
    assert [answer.status_code for answer in answers] == [200] * 30
    assert over.status_code == 429
    assert over.json()["retry_after"] == 40
    assert over.headers["retry-after"] == "40"
    assert isinstance(over.json()["detail"], str)
    assert asked == 60  # The 31st call fetched nothing: robots.txt and the feed, 30 times.
    assert again.status_code == 200


def test_a_refused_call_does_not_count_against_the_cap(tmp_path, web, clock):
    with api(tmp_path, web, clock) as client:
        for _ in range(40):
            client.post("/api/feeds/check", json={"url": GOOD})  # No token: 401.
        response = client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
    assert response.status_code == 200


# The re-check before a run.


def test_a_run_with_2_good_and_1_bad_feed_starts_with_the_good_ones(tmp_path, web):
    runner = FakeRunner()
    with api(tmp_path, web, runner=runner) as client:
        response = client.post(
            "/api/runs", json={"feeds": [GOOD, PAGE, ALSO_GOOD]}, headers=headers()
        )
        assert response.status_code == 202, response.text
        assert client.app.state.jobs.wait(response.json()["id"], 5)
    (request,) = runner.requests
    assert request.feeds == [GOOD, ALSO_GOOD]
    assert request.failed_feeds == [FeedFailure(PAGE, fc.NOT_A_FEED)]


def test_a_run_lists_the_bad_feed_in_its_fetching_feeds_step(tmp_path):
    from test_cli import ENV, NEWS, by_meaning, model, run_web

    def run_and_page(url, headers, timeout):
        return (
            Web()(url, headers, timeout) if url in (PAGE, DOWN) else run_web(url, headers, timeout)
        )

    runner = default_runner(ENV, transport=model(embed=by_meaning), fetch=run_and_page)
    with api(tmp_path, run_and_page, runner=runner) as client:
        response = client.post("/api/runs", json={"feeds": [NEWS, PAGE, DOWN]}, headers=headers())
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        assert client.app.state.jobs.wait(run_id, 10)
        run = client.get(f"/api/runs/{run_id}", headers=headers()).json()
        draft = client.get(f"/api/drafts/{run['draft_id']}", headers=headers()).json()
    assert run["status"] == "done", run
    feeds = next(step for step in run["progress"] if step["label"] == "Fetching feeds")
    assert feeds["state"] == "done"
    assert (feeds["done"], feeds["total"]) == (3, 3)
    assert feeds["detail"] == (
        f"1 of 3 loaded, 2 failed: {PAGE} {fc.NOT_A_FEED}; {DOWN} {fc.UNREACHABLE}"
    )
    failed = {(f["url"], f["reason"]) for f in draft["report"]["failed_sources"]}
    assert {(PAGE, fc.NOT_A_FEED), (DOWN, fc.UNREACHABLE)} <= failed


def test_failed_feeds_open_the_fetching_feeds_step_and_are_in_the_failures():
    events = []

    def progress(step, state, **counts):
        events.append((step, state, counts))

    class NoEmbedding:
        def embed(self, step, inputs):
            raise AssertionError("no item, no embedding call")

    gathered = gather_stories(
        [],
        llm=NoEmbedding(),
        topic="news",
        failed=[FeedFailure(PAGE, fc.NOT_A_FEED)],
        fetch=Web(),
        progress=progress,
    )
    assert gathered.failures == [FeedFailure(PAGE, fc.NOT_A_FEED)]
    first = events[0]
    assert first == (
        "feeds",
        "running",
        {"detail": f"0 of 1 loaded, 1 failed: {PAGE} {fc.NOT_A_FEED}", "done": 1, "total": 1},
    )
    gather_stories([], llm=NoEmbedding(), topic="news", fetch=Web(), progress=no_progress)


def test_a_run_whose_feeds_all_fail_is_422_with_an_error_per_feed(tmp_path, web):
    runner = FakeRunner()
    with api(tmp_path, web, runner=runner) as client:
        response = client.post("/api/runs", json={"feeds": [PAGE, EMPTY, SHY]}, headers=headers())
        listed = client.get("/api/drafts", headers=headers()).json()
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], str)
    assert body["errors"] == [
        {"url": PAGE, "error": fc.NOT_A_FEED},
        {"url": EMPTY, "error": fc.NO_ITEMS},
        {"url": SHY, "error": fc.ROBOTS},
    ]
    assert runner.requests == []
    assert listed == []


def test_a_run_refuses_a_loopback_feed_like_the_check(tmp_path, web):
    runner = FakeRunner()
    with api(tmp_path, web, runner=runner) as client:
        response = client.post(
            "/api/runs", json={"feeds": ["http://localhost/feed.xml"]}, headers=headers()
        )
    assert response.status_code == 422
    assert response.json()["errors"] == [{"url": "http://localhost/feed.xml", "error": fc.PRIVATE}]


def test_a_run_from_the_sources_config_is_not_rechecked(tmp_path, web):
    from test_api import CONFIG

    runner = FakeRunner()
    settings = make_settings(tmp_path)
    app = create_app(settings, runner=runner, config=CONFIG, fetch=web, resolve=public)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"title": "Weekly"}, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
    assert response.status_code == 202
    assert web.asked == []
    assert runner.requests[0].feeds == list(CONFIG.feeds)


def test_the_recheck_does_not_count_against_the_checks_cap(tmp_path, web, clock):
    with api(tmp_path, web, clock) as client:
        for _ in range(29):
            client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
        started = client.post("/api/runs", json={"feeds": [GOOD, ALSO_GOOD]}, headers=headers())
        assert client.app.state.jobs.wait(started.json()["id"], 5)
        last = client.post("/api/feeds/check", json={"url": GOOD}, headers=headers())
    assert started.status_code == 202
    assert last.status_code == 200


# Busy.


def test_a_second_run_while_one_runs_is_409_with_retry_after(tmp_path, web):
    release = threading.Event()
    with api(tmp_path, web, runner=FakeRunner(block=release)) as client:
        first = client.post("/api/runs", json={"feeds": [GOOD]}, headers=headers())
        asked = len(web.asked)
        second = client.post("/api/runs", json={"feeds": [ALSO_GOOD]}, headers=headers())
        release.set()
        assert client.app.state.jobs.wait(first.json()["id"], 5)
    assert first.status_code == 202
    assert second.status_code == 409
    body = second.json()
    assert set(body) == {"detail", "retry_after"}
    assert isinstance(body["retry_after"], int) and body["retry_after"] > 0
    assert second.headers["retry-after"] == str(body["retry_after"])
    assert len(web.asked) == asked  # A busy server checks no feed.


# Demo mode.


def test_demo_mode_is_off_unless_newsletter_demo_is_1():
    assert ApiSettings.from_env({}).demo is False
    assert ApiSettings.from_env({"NEWSLETTER_DEMO": "0"}).demo is False
    assert ApiSettings.from_env({"NEWSLETTER_DEMO": ""}).demo is False
    assert ApiSettings.from_env({"NEWSLETTER_DEMO": "1"}).demo is True


def seven(web_feeds=GOOD) -> list[str]:
    return [f"{web_feeds}?n={n}" for n in range(7)]


class AnyGood(Web):
    def __call__(self, url, headers, timeout):
        return super().__call__(url.split("?")[0], headers, timeout)


def test_a_7th_feed_is_422_in_demo_mode(tmp_path):
    web = AnyGood()
    runner = FakeRunner()
    with api(tmp_path, web, runner=runner, NEWSLETTER_DEMO="1") as client:
        response = client.post("/api/runs", json={"feeds": seven()}, headers=headers())
    assert response.status_code == 422
    assert "6" in str(response.json()["detail"])
    assert runner.requests == []
    assert web.asked == []


def test_six_feeds_run_in_demo_mode(tmp_path):
    runner = FakeRunner()
    with api(tmp_path, AnyGood(), runner=runner, NEWSLETTER_DEMO="1") as client:
        response = client.post("/api/runs", json={"feeds": seven()[:6]}, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
    assert response.status_code == 202
    assert len(runner.requests[0].feeds) == 6


@pytest.mark.parametrize("asked,used", [(None, 3), (1, 1), (3, 3), (10, 3)])
def test_demo_mode_reads_at_most_3_items_per_feed(tmp_path, web, asked, used):
    runner = FakeRunner()
    body = {"feeds": [GOOD]} if asked is None else {"feeds": [GOOD], "per_feed": asked}
    with api(tmp_path, web, runner=runner, NEWSLETTER_DEMO="1") as client:
        response = client.post("/api/runs", json=body, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
    assert runner.requests[0].per_feed == used


def test_demo_mode_ignores_the_date(tmp_path, web):
    runner = FakeRunner()
    with api(tmp_path, web, runner=runner, NEWSLETTER_DEMO="1") as client:
        body = {"feeds": [GOOD], "date": "2020-01-01"}
        response = client.post("/api/runs", json=body, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
    assert response.status_code == 202
    assert runner.requests[0].issue_date is None


def test_with_demo_mode_off_the_old_request_shapes_work(tmp_path):
    runner = FakeRunner()
    web = AnyGood()
    body = {"feeds": seven(), "title": "Weekly", "topic": "AI", "date": "2026-11-02"}
    with api(tmp_path, web, runner=runner) as client:
        response = client.post("/api/runs", json=body, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
        per_feed = client.post(
            "/api/runs", json={"feeds": [GOOD], "per_feed": 10}, headers=headers()
        )
        assert client.app.state.jobs.wait(per_feed.json()["id"], 5)
    assert response.status_code == 202, response.text
    first, second = runner.requests
    assert len(first.feeds) == 7
    assert first.issue_date == dt.date(2026, 11, 2)
    assert first.per_feed is None
    assert second.per_feed == 10


def test_without_demo_mode_up_to_50_feeds_are_taken(tmp_path):
    runner = FakeRunner()
    feeds = [f"{GOOD}?n={n}" for n in range(51)]
    with api(tmp_path, AnyGood(), runner=runner) as client:
        response = client.post("/api/runs", json={"feeds": feeds}, headers=headers())
    assert response.status_code == 422
    assert runner.requests == []


# Presets.


def test_presets_list_ai_news_science_and_tech_each_with_3_or_4_public_feeds(tmp_path, web):
    with api(tmp_path, web) as client:
        response = client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()
    assert [p["id"] for p in presets] == ["ai", "science", "tech"]
    for preset in presets:
        assert set(preset) == {"id", "label", "feeds"}
        assert preset["label"]
        assert 3 <= len(preset["feeds"]) <= 4
        assert all(url.startswith("https://") for url in preset["feeds"])
    assert web.asked == []


# The spec.


def test_the_new_routes_and_the_retry_after_answers_are_in_the_spec(tmp_path, web):
    with api(tmp_path, web) as client:
        spec = client.get("/api/openapi.json").json()
    check = spec["paths"]["/api/feeds/check"]["post"]
    assert {"200", "401", "422", "429"} <= set(check["responses"])
    assert "Retry-After" in check["responses"]["429"]["headers"]
    assert "/api/presets" in spec["paths"]
    run = spec["paths"]["/api/runs"]["post"]["responses"]
    assert "Retry-After" in run["409"]["headers"]
    retry = spec["components"]["schemas"]["RetryLater"]
    assert set(retry["properties"]) == {"detail", "retry_after"}
    assert "errors" in spec["components"]["schemas"]["InvalidRequest"]["properties"]
    assert set(spec["components"]["schemas"]["FeedCheckView"]["properties"]) == {
        "ok",
        "title",
        "items",
        "error",
    }
    assert TOKEN not in str(spec)


def test_demo_mode_keeps_a_smaller_per_feed_from_the_sources_config(tmp_path, web):
    from test_api import CONFIG  # per_feed=2

    runner = FakeRunner()
    settings = make_settings(tmp_path, NEWSLETTER_DEMO="1")
    app = create_app(settings, runner=runner, config=CONFIG, fetch=web, resolve=public)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"feeds": [GOOD]}, headers=headers())
        assert client.app.state.jobs.wait(response.json()["id"], 5)
    assert runner.requests[0].per_feed == 2


def test_the_feed_errors_hold_no_secret(tmp_path, web):
    leaky = f"https://page.example.org/?key={TOKEN}"

    class Leaky(Web):
        def __call__(self, url, headers, timeout):
            return super().__call__(PAGE if url == leaky else url, headers, timeout)

    with api(tmp_path, Leaky()) as client:
        response = client.post("/api/runs", json={"feeds": [leaky]}, headers=headers())
    assert response.status_code == 422
    assert TOKEN not in response.text
    assert response.json()["errors"][0]["error"] == fc.NOT_A_FEED


def test_an_unexpected_failure_reading_the_address_is_a_sentence_not_a_crash(caplog):
    def broken(url, headers, timeout):
        raise ValueError("something nobody planned for")

    result = fc.check_feed(GOOD, fetch=broken, resolve=public)
    assert result == fc.FeedCheck(ok=False, title=None, items=0, error=fc.UNREACHABLE)
    assert "something nobody planned for" in caplog.text


def test_an_unreadable_robots_txt_means_the_address_did_not_answer():
    def web(url, headers, timeout):
        return HttpResponse(503) if url.endswith("/robots.txt") else HttpResponse(200, {}, RSS)

    assert fc.check_feed(GOOD, fetch=web, resolve=public).error == fc.UNREACHABLE


def test_a_run_that_loses_the_race_to_start_is_409_with_retry_after(tmp_path, web):
    from newsletter.api.jobs import Busy

    with api(tmp_path, web) as client:

        def lost(request):
            raise Busy("a run is already going")

        client.app.state.jobs.start = lost
        response = client.post("/api/runs", json={"feeds": [GOOD]}, headers=headers())
    assert response.status_code == 409
    assert response.json()["retry_after"] > 0
    assert response.headers["retry-after"] == str(response.json()["retry_after"])
