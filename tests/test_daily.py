"""The daily archive workflow, and the same build run here with fake feeds and a fake model."""

import shlex
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
import yaml

from newsletter.cli import main
from newsletter.sources.http import HttpResponse

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "daily.yml"
ATOM = (Path(__file__).parent / "fixtures" / "feeds" / "atom.xml").read_bytes()
MODEL_HOST = "model.test"
ENV = {
    "NEWSLETTER_BASE_URL": f"http://{MODEL_HOST}/v1",
    "NEWSLETTER_FAST_MODEL": "fast-1",
    "NEWSLETTER_MAX_RETRIES": "0",
}
DEPLOY_ACTIONS = ("deploy-pages", "upload-pages-artifact", "gh-pages", "pages-deploy", "deploy")
PUBLISH_COMMANDS = ("git push", "gh ", "deploy", "rsync", "scp ", "curl ", "wget ", "aws ", "s3")


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def triggers(workflow):
    # YAML 1.1 reads a bare `on` key as True.
    return workflow.get("on", workflow.get(True))


def steps(workflow):
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def build_step(workflow):
    (step,) = [step for step in steps(workflow) if step.get("id") == "build"]
    return step


def build_argv(workflow) -> list[str]:
    argv = shlex.split(build_step(workflow)["run"])
    assert argv[:3] == ["uv", "run", "newsletter"], argv
    return argv[3:]


def test_runs_by_workflow_dispatch_only(workflow):
    assert set(triggers(workflow)) == {"workflow_dispatch"}
    assert "pull_request_target" not in WORKFLOW.read_text(encoding="utf-8")


def test_every_job_has_a_timeout(workflow):
    for name, job in workflow["jobs"].items():
        assert isinstance(job.get("timeout-minutes"), int), name
        assert 0 < job["timeout-minutes"] <= 60, name


def test_the_token_can_only_read_the_repo(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in workflow["jobs"].items():
        assert job.get("permissions", {"contents": "read"}) == {"contents": "read"}, name


def test_there_is_no_deploy_step(workflow):
    for step in steps(workflow):
        uses = step.get("uses", "").lower()
        assert not any(action in uses for action in DEPLOY_ACTIONS), uses
        assert "deploy" not in step.get("name", "").lower(), step
        run = step.get("run", "").lower()
        assert not any(command in run for command in PUBLISH_COMMANDS), run


def test_checkout_does_not_keep_the_token(workflow):
    (checkout,) = [s for s in steps(workflow) if s.get("uses", "").startswith("actions/checkout")]
    assert checkout["with"]["persist-credentials"] is False


def test_the_build_is_the_demo_archive_from_public_https_feeds(workflow):
    argv = build_argv(workflow)
    assert argv[0] == "archive"
    assert "--demo" in argv
    feeds = [arg for arg in argv[1:] if "://" in arg]
    assert len(feeds) >= 3
    assert all(urlsplit(feed).scheme == "https" for feed in feeds)


def test_the_site_is_uploaded_as_a_workflow_artifact(workflow):
    argv = build_argv(workflow)
    site = argv[argv.index("--site") + 1]
    (upload,) = [s for s in steps(workflow) if s.get("uses", "").startswith("actions/upload-arti")]
    assert upload["with"]["path"].rstrip("/") == site
    assert upload["with"]["if-no-files-found"] == "error"
    assert "always()" not in upload.get("if", ""), "a cancelled run uploads nothing"
    assert steps(workflow).index(upload) > steps(workflow).index(build_step(workflow))


def test_the_model_key_comes_from_a_secret_and_nothing_else_does(workflow):
    env = build_step(workflow)["env"]
    assert env["NEWSLETTER_API_KEY"] == "${{ secrets.NEWSLETTER_API_KEY }}"
    secrets = [key for key, value in env.items() if "secrets." in str(value)]
    assert secrets == ["NEWSLETTER_API_KEY"]


def test_the_same_build_runs_with_a_fake_model_and_publishes_nothing(
    workflow, tmp_path, monkeypatch
):
    argv = build_argv(workflow)
    feeds = {arg for arg in argv if "://" in arg}
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
    code = main(argv, env=ENV, fetch=fetch, llm_transport=httpx.MockTransport(model))

    assert code == 0
    site = tmp_path / argv[argv.index("--site") + 1]
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


def test_the_daily_build_caps_its_model_calls(workflow):
    argv = build_argv(workflow)
    feeds = [arg for arg in argv if "://" in arg]
    per_feed = int(argv[argv.index("--per-feed") + 1])
    assert len(feeds) * per_feed <= 50
