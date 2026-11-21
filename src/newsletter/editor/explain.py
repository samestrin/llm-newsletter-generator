"""Why a claim is flagged, in one or two plain sentences built in code.

The numbers are read with the voice check's extractor (`extract_facts`). When a number
of the claim is missing from its passage and the passage has a different number near
the same words, the sentence says both: "The article says 71; the draft said 95."
Otherwise it says the check's status in plain words, then the checking model's reason,
marked as the model's. Every figure in the sentence must be in the claim or the
passage; when one is not (a model reason can say anything), the status is said alone.
"""

import re

from newsletter.editor.faithfulness import Status
from newsletter.voice.check import extract_facts

PLAIN = {
    "unsupported": "The article does not say this.",
    "unchecked": "This claim could not be checked.",
}

_WINDOW = 3
"""How many words on each side of a number are its context."""
_MIN_SHARED = 2
"""How many context words a passage number must share to answer a claim number."""
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this
    to was were with""".split()
)
_WORD = re.compile(r"[a-z]+")


def explain(claim: str, passage: str, status: Status, reason: str) -> str:
    """The explanation of a flagged claim. `reason` is the check's: the model's reason
    when it said no, or why the check failed."""
    plain = PLAIN.get(status, PLAIN["unchecked"])
    sentence = _changed_numbers(claim, passage)
    if sentence is None:
        sentence = plain
        # An unchecked claim's reason is an error, not the model's word.
        if status == "unsupported" and reason.strip():
            sentence += f' The checking model\'s reason: "{reason.strip()}"'
    if all(n in claim or n in passage for n in extract_facts(sentence).numbers):
        return sentence
    return plain


def _changed_numbers(claim: str, passage: str) -> str | None:
    """One "The article says X; the draft said Y." for each number of the claim missing
    from the passage that a different number of the passage answers, or None if none does.

    A passage number answers a claim number when at least `_MIN_SHARED` of the words
    just before and just after them are the same; each answers one claim number at most,
    the closest match first."""
    claim_numbers, passage_numbers = _numbers(claim), _numbers(passage)
    in_passage = {number for number, _, _ in passage_numbers}
    in_claim = {number for number, _, _ in claim_numbers}
    missing = [(i, n) for i, n in enumerate(claim_numbers) if n[0] not in in_passage]
    others = [(j, n) for j, n in enumerate(passage_numbers) if n[0] not in in_claim]
    candidates = []
    for i, (_, left, right) in missing:
        for j, (_, p_left, p_right) in others:
            shared = len(left & p_left) + len(right & p_right)
            if shared >= _MIN_SHARED:
                candidates.append((-shared, abs(i - j), i, j))
    pairs: dict[int, int] = {}
    for _, _, i, j in sorted(candidates):
        if i not in pairs and j not in pairs.values():
            pairs[i] = j
    if not pairs:
        return None
    return " ".join(
        f"The article says {passage_numbers[j][0]}; the draft said {claim_numbers[i][0]}."
        for i, j in sorted(pairs.items())
    )


def _numbers(text: str) -> list[tuple[str, frozenset[str], frozenset[str]]]:
    """Each number of `text` in order, with the content words just before and after it."""
    words: list[str] = []
    found: list[tuple[str, int]] = []
    at = 0
    for number in extract_facts(text).numbers:
        start = text.find(number, at)
        if start < 0:
            continue
        words += _content_words(text[at:start])
        found.append((number, len(words)))
        at = start + len(number)
    words += _content_words(text[at:])
    return [
        (
            number,
            frozenset(words[max(0, position - _WINDOW) : position]),
            frozenset(words[position : position + _WINDOW]),
        )
        for number, position in found
    ]


def _content_words(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]
