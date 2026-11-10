"""The committed sample in `docs/sample/`: its README records the model, the date, the
feeds and the one command, and every file in it is labeled as an automated demo.

The sample is generated once by the owner with a real model and committed; CI never
regenerates it. Until then the README's date reads `pending` and only the README is
there; once the date is set, every file the command writes must be there.
"""

import re
import shlex
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from newsletter.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "docs" / "sample"
README = SAMPLE / "README.md"
LABEL = "Automated demo"
MODELS = ("kimi-k3", "glm-5.3")  # The owner's choice; glm-5.3 only with high thinking effort.
PENDING = "pending"


def field(name: str) -> str:
    match = re.search(rf"^- \*\*{name}:\*\* (.+)$", README.read_text(encoding="utf-8"), re.M)
    assert match, f"README records no {name}"
    return match.group(1).strip().strip("`")


def recorded_feeds() -> list[str]:
    text = README.read_text(encoding="utf-8")
    block = re.search(r"^- \*\*Feeds:\*\*\n((?:  - .+\n)+)", text, re.M)
    assert block, "README records no feed list"
    return [line.removeprefix("  - ").strip() for line in block.group(1).splitlines()]


def command() -> tuple[dict[str, str], list[str]]:
    """The environment and the `newsletter` argv of the recorded command."""
    text = README.read_text(encoding="utf-8")
    (block,) = re.findall(r"^```sh\n(.*?)^```$", text, re.M | re.S)
    words = shlex.split(block.replace("\\\n", " "))
    env = {}
    while "=" in words[0]:
        name, value = words.pop(0).split("=", 1)
        env[name] = value
    assert words[:3] == ["uv", "run", "newsletter"], words
    return env, words[3:]


def recorded_date() -> date | None:
    value = field("Date")
    return None if value == PENDING else date.fromisoformat(value)


def written_files() -> list[Path]:
    """The files the recorded command writes, relative to the repo root."""
    _, argv = command()
    args = build_parser().parse_args(argv)
    day = recorded_date()
    assert day is not None
    site = Path(args.site)
    return [
        Path(args.markdown),
        Path(args.html),
        site / "index.html",
        site / "issues" / day.isoformat() / "index.html",
    ]


def present() -> list[Path]:
    return sorted(path for path in SAMPLE.rglob("*") if path.is_file())


def test_the_readme_records_the_model_date_feeds_and_command():
    assert field("Model") in MODELS
    value = field("Date")
    assert value == PENDING or date.fromisoformat(value)
    feeds = recorded_feeds()
    assert feeds, "the recorded feed list is non-empty"
    assert all(urlsplit(feed).scheme == "https" for feed in feeds)
    _, argv = command()
    assert argv


def test_the_command_is_the_demo_archive_of_the_recorded_feeds_and_model():
    env, argv = command()
    args = build_parser().parse_args(argv)
    assert args.command == "archive" and args.demo
    assert args.feeds == recorded_feeds()
    assert env["NEWSLETTER_FAST_MODEL"] == field("Model")
    assert env["NEWSLETTER_REASONING_MODEL"] == field("Model")
    assert not any("KEY" in name or "TOKEN" in name for name in env), "no key in the README"
    for path in (args.site, args.markdown, args.html):
        assert Path(path).resolve().is_relative_to(SAMPLE.resolve()), path


def test_every_file_in_the_sample_carries_the_automated_demo_label():
    for path in present():
        assert LABEL in path.read_text(encoding="utf-8"), path.relative_to(ROOT)


def test_the_files_are_the_ones_the_command_writes():
    if recorded_date() is None:
        # Not generated yet: nothing but the README, so no stale or hand-made sample.
        assert present() == [README]
        return
    expected = {(ROOT / path).resolve() for path in written_files()}
    for path in expected:
        assert path.is_file(), path.relative_to(ROOT)
    assert {path.resolve() for path in present()} == expected | {README.resolve()}


def test_the_pages_open_offline():
    for path in present():
        if path.suffix != ".html":
            continue
        parser = Loads()
        parser.feed(path.read_text(encoding="utf-8"))
        assert not parser.loads, (path.relative_to(ROOT), parser.loads)


class Loads(HTMLParser):
    """Everything a page would fetch to show itself: images, scripts, stylesheets, frames."""

    def __init__(self):
        super().__init__()
        self.loads = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.loads += [values[name] for name in ("src", "srcset") if values.get(name)]
        if tag == "link" and "stylesheet" in (values.get("rel") or ""):
            self.loads.append(values.get("href"))
