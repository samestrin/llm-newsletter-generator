"""The demo archive build, run here with fake feeds and a fake model."""

from pathlib import Path
from urllib.parse import urlsplit

import httpx

from newsletter.cli import main
from newsletter.sources.http import HttpResponse

ATOM = (Path(__file__).parent / "fixtures" / "feeds" / "atom.xml").read_bytes()
MODEL_HOST = "model.test"
ENV = {
    "NEWSLETTER_BASE_URL": f"http://{MODEL_HOST}/v1",
    "NEWSLETTER_FAST_MODEL": "fast-1",
    "NEWSLETTER_MAX_RETRIES": "0",
}
ARGV = [
    "archive",
    "--demo",
    "--site",
    "site",
    "--title",
    "AI Daily (automated demo)",
    "--topic",
    "AI",
    "--per-feed",
    "5",
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://huggingface.co/blog/feed.xml",
]


def test_the_same_build_runs_with_a_fake_model_and_publishes_nothing(tmp_path, monkeypatch):
    feeds = {arg for arg in ARGV if "://" in arg}
    fetched, called = [], []

    def fetch(url, headers, timeout):
        fetched.append(url)
        if url in feeds:
            return HttpResponse(200, {}, ATOM)
        return HttpResponse(404)

    def model(request):
        called.append(request.url.host)
        return httpx.Response(200, json={"choices": [{"message": {"content": "A summary."}}]})

    monkeypatch.chdir(tmp_path)
    code = main(ARGV, env=ENV, fetch=fetch, llm_transport=httpx.MockTransport(model))

    assert code == 0
    site = tmp_path / ARGV[ARGV.index("--site") + 1]
    index = (site / "index.html").read_text(encoding="utf-8")
    (page,) = (site / "issues").glob("*/index.html")
    text = page.read_text(encoding="utf-8")
    for html in (index, text):
        assert "Automated demo" in html
    assert "Flagged: unchecked" in text and "A summary." in text
    # Without publishing: the build read only its own feeds and the model, and wrote
    # only the site.
    assert {urlsplit(url).hostname for url in fetched} <= {urlsplit(f).hostname for f in feeds}
    assert set(called) == {MODEL_HOST}
    assert sorted(p.name for p in tmp_path.iterdir()) == [site.name]
