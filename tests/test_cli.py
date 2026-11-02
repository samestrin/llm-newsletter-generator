import pytest

from newsletter.cli import build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "newsletter" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert "newsletter 0.0.4" in capsys.readouterr().out


def test_main_returns_zero():
    assert main([]) == 0
