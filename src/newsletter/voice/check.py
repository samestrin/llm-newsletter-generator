"""The fact check on a voice rewrite: the facts in the prose must survive it unchanged.

A rewrite may change the words around the facts, but every link, quoted string, number
and claim-id reference of the neutral text must still be there, exactly and as often,
and nothing of those kinds may be added. Anything else is a problem, and the caller
keeps the neutral text.
"""

import re
from collections import Counter
from dataclasses import dataclass

_URL = re.compile(r"https?://[^\s<>()\[\]\"“”]+")
_URL_TRAILING = ".,;:!?'"
_CLAIM_ID = re.compile(r"\b[A-Za-z][\w-]*\.c\d+\b")
_QUOTE = re.compile(r"\"([^\"\n]+)\"|“([^”\n]+)”")
# Digits joined by . , : - or / are one number, so "1,000" and "2026-11-02" stay whole.
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,:/-]\d+)*%?")


@dataclass(frozen=True)
class Facts:
    """What a rewrite must keep, each in the order it appears in the text."""

    urls: tuple[str, ...]
    quotes: tuple[str, ...]
    numbers: tuple[str, ...]
    claim_ids: tuple[str, ...]


def extract_facts(text: str) -> Facts:
    urls = tuple(match.group().rstrip(_URL_TRAILING) for match in _URL.finditer(text))
    # Each kind is cut out before the next is read, so the 42 in a URL or the 1 in
    # `s1.c1` is not counted again as a number.
    rest = _URL.sub(" ", text)
    claim_ids = tuple(_CLAIM_ID.findall(rest))
    rest = _CLAIM_ID.sub(" ", rest)
    quotes = tuple(match.group(1) or match.group(2) for match in _QUOTE.finditer(rest))
    numbers = tuple(_NUMBER.findall(rest))
    return Facts(urls, quotes, numbers, claim_ids)


def compare_facts(before: str, after: str) -> list[str]:
    """Every fact the rewrite dropped, changed or added; empty when it kept them all."""
    old, new = extract_facts(before), extract_facts(after)
    problems = []
    for kind, field_name, show in (
        ("link", "urls", str),
        ("quote", "quotes", lambda quote: f'"{quote}"'),
        ("number", "numbers", str),
        ("claim id", "claim_ids", str),
    ):
        kept, written = Counter(getattr(old, field_name)), Counter(getattr(new, field_name))
        problems += [f"{kind} {show(fact)} was dropped" for fact in kept - written]
        problems += [f"{kind} {show(fact)} was added" for fact in written - kept]
    return problems
