from pathlib import Path

import pytest

from newsletter.config import SourcesConfig, SourcesConfigError, load_config
from test_cli import ATOM, RSS, run

VALID = """\
topic: open models
sources:
  - https://example.com/feed.xml
  - https://lab.example.org/atom.xml
limits:
  per_feed: 2
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_valid_file(tmp_path):
    config = load_config(write(tmp_path, VALID))
    assert config == SourcesConfig(
        topic="open models",
        sources=("https://example.com/feed.xml", "https://lab.example.org/atom.xml"),
        per_feed=2,
    )


def test_limits_are_optional(tmp_path):
    config = load_config(write(tmp_path, "topic: t\nsources: [https://a.test/f]\n"))
    assert config.per_feed is None


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("sources: [https://a.test/f]\n", "topic"),
        ("topic: ''\nsources: [https://a.test/f]\n", "topic"),
        ("topic: 3\nsources: [https://a.test/f]\n", "topic"),
        ("topic: t\n", "sources"),
        ("topic: t\nsources: []\n", "sources"),
        ("topic: t\nsources: https://a.test/f\n", "sources"),
        ("topic: t\nsources: [3]\n", "sources[0]"),
        ("topic: t\nsources: [https://a.test/f, 'ftp://x.test/f']\n", "sources[1]"),
        ("topic: t\nsources: [https://a.test/f]\nlimits: 3\n", "limits"),
        ("topic: t\nsources: [https://a.test/f]\nlimits: {per_feed: 0}\n", "limits.per_feed"),
        ("topic: t\nsources: [https://a.test/f]\nlimits: {per_feed: x}\n", "limits.per_feed"),
        ("topic: t\nsources: [https://a.test/f]\nlimits: {per_feed: true}\n", "limits.per_feed"),
        ("topic: t\nsources: [https://a.test/f]\nlimits: {perfeed: 1}\n", "limits.perfeed"),
        ("topic: t\nsources: [https://a.test/f]\nextra: 1\n", "extra"),
        ("- a\n- b\n", "config"),
        ("topic: [unclosed\n", "config"),
    ],
)
def test_invalid_config_names_the_field(tmp_path, text, field):
    path = write(tmp_path, text)
    with pytest.raises(SourcesConfigError) as exc:
        load_config(path)
    assert field in str(exc.value)
    assert str(path) in str(exc.value)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(SourcesConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")


def test_cli_loads_the_config(tmp_path, capsys):
    path = write(tmp_path, f"topic: open models\nsources: [{ATOM}]\n")
    code = run(["archive", "--config", str(path), "--demo", "--site", str(tmp_path / "site")])
    assert code == 0
    assert "1 story" in capsys.readouterr().err


def test_config_per_feed_limit_applies_and_the_flag_wins(tmp_path, capsys):
    path = write(tmp_path, f"topic: t\nsources: [{RSS}]\nlimits: {{per_feed: 1}}\n")
    site = str(tmp_path / "site")
    run(["archive", "--config", str(path), "--demo", "--site", site])
    assert "1 story" in capsys.readouterr().err
    run(["archive", "--config", str(path), "--per-feed", "2", "--demo", "--site", site])
    assert "2 stories" in capsys.readouterr().err


def test_cli_feeds_win_over_the_config(tmp_path, capsys):
    path = write(tmp_path, f"topic: t\nsources: [{RSS}]\n")
    run(["archive", ATOM, "--config", str(path), "--demo", "--site", str(tmp_path / "site")])
    assert "1 story" in capsys.readouterr().err


def test_cli_reports_an_invalid_config_with_exit_2(tmp_path, capsys):
    path = write(tmp_path, "topic: t\nsources: []\n")
    assert run(["issue", "--config", str(path)]) == 2
    assert "sources" in capsys.readouterr().err
