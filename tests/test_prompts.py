"""Snapshots of the prompt templates that ship with the package.

A prompt is product behaviour: changing one changes what a model is asked, so every
template is pinned to the text under `tests/snapshots/prompts/`. A prompt change and its
snapshot belong in the same commit, which makes an accidental edit show up in review.

The tests also pin the prompt *set* and the names the code loads, so a removed prompt
cannot linger and a new one cannot ship without a snapshot.
"""

import re
from importlib.resources import files
from pathlib import Path

import pytest

from newsletter.issue import load_prompt

PACKAGE = files("newsletter")
SOURCE = Path(__file__).resolve().parents[1] / "src" / "newsletter"
SNAPSHOTS = Path(__file__).parent / "snapshots" / "prompts"
PROMPTS = sorted(
    entry.name for entry in PACKAGE.joinpath("prompts").iterdir() if entry.name.endswith(".md")
)
REMOVED = {"item", "introduction", "closing"}
"""The epic 01 T4 prompts that epic 03 T4 replaces; they must not come back."""

_LOAD_PROMPT = re.compile(r'load_prompt\(\s*"([^"]+)"\s*\)')


def prompt_names_in_code() -> set[str]:
    """Every name passed to `load_prompt` by the package's source."""
    names: set[str] = set()
    for path in SOURCE.rglob("*.py"):
        names.update(_LOAD_PROMPT.findall(path.read_text(encoding="utf-8")))
    return names


def test_every_prompt_ships_with_a_snapshot():
    snapshots = sorted(path.name for path in SNAPSHOTS.glob("*.md"))

    assert snapshots == PROMPTS


@pytest.mark.parametrize("name", PROMPTS)
def test_prompt_matches_its_snapshot(name):
    assert load_prompt(name.removesuffix(".md")) == (SNAPSHOTS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", PROMPTS)
def test_prompt_is_not_empty(name):
    assert load_prompt(name.removesuffix(".md")).strip()


def test_every_prompt_the_code_loads_ships_with_a_snapshot():
    assert {f"{name}.md" for name in prompt_names_in_code()} <= set(PROMPTS)


def test_the_removed_prompts_are_gone_and_not_referenced():
    shipped = {name.removesuffix(".md") for name in PROMPTS}

    assert shipped.isdisjoint(REMOVED)
    assert prompt_names_in_code().isdisjoint(REMOVED)
