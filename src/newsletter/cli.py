"""Command-line entry point for the `newsletter` command."""

import argparse
from collections.abc import Sequence

from newsletter import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsletter",
        description="Generate text-only newsletters from RSS feeds using LLMs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
