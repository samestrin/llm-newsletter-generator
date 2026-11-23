"""No test reaches the network through the API's feed check.

`create_app` checks a run's feeds with `http_get` and `resolve_host` unless a test passes
its own `fetch` and `resolve`. Here every host is public and every address answers a
one-item feed, so a test about something else starts its run as before; a test about the
check passes its own fake web.
"""

import pytest

from newsletter.sources.http import HttpResponse

ANY_FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Any feed</title>
<item><title>A story</title><link>https://example.org/a-story</link></item>
</channel></rss>"""


def any_feed(url, headers, timeout):
    if url.endswith("/robots.txt"):
        return HttpResponse(404)
    return HttpResponse(200, {}, ANY_FEED)


@pytest.fixture(autouse=True)
def offline_feed_check(monkeypatch):
    monkeypatch.setattr("newsletter.api.app.http_get", any_feed)
    monkeypatch.setattr("newsletter.api.app.resolve_host", lambda host: ["93.184.216.34"])
