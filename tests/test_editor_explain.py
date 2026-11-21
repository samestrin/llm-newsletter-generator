import json
from pathlib import Path

import pytest

from newsletter.editor.explain import explain
from newsletter.voice.check import extract_facts

CASES = Path(__file__).parent.parent / "src" / "newsletter" / "evals" / "cases"

# Hand-made pairs: (claim, passage, status, reason, expected sentence).
CHANGED_NUMBER = (
    "Tiny-3B scores 95 on the MMLU benchmark.",
    "It scores 71 on the MMLU benchmark and runs on a laptop with 16 GB of memory.",
    "unsupported",
    "The passage gives 71, not 95.",
    "The article says 71; the draft said 95.",
)
ADDED_NUMBER = (
    "The new chip uses 40% less power than the old one.",
    "The new chip uses less power than the old one.",
    "unsupported",
    "The passage gives no figure for the saving.",
    "The article does not say this. The checking model's reason: \"The passage gives no "
    'figure for the saving."',
)
NO_NUMBERS = (
    "The result proves the planet is habitable.",
    "The team published the result in a peer reviewed journal on Thursday.",
    "unsupported",
    "The passage says nothing about habitability.",
    "The article does not say this. The checking model's reason: \"The passage says nothing "
    'about habitability."',
)


@pytest.mark.parametrize(
    "claim, passage, status, reason, expected",
    [CHANGED_NUMBER, ADDED_NUMBER, NO_NUMBERS],
    ids=["changed-number", "added-number", "no-numbers"],
)
def test_hand_made_pairs_give_the_expected_sentence(claim, passage, status, reason, expected):
    assert explain(claim, passage, status, reason) == expected


def _recorded_flags() -> list[tuple[str, str, str, str]]:
    """(case/claim id, claim, passage, reason) for every claim a recorded check flags."""
    flags = []
    for path in sorted(CASES.glob("*/recorded.json")):
        responses = json.loads(path.read_text(encoding="utf-8"))["responses"]
        claims = {}
        for step, answer in responses.items():
            if step.startswith("claims-") and isinstance(answer.get("content"), dict):
                for number, raw in enumerate(answer["content"]["claims"], start=1):
                    claims[f"{step.removeprefix('claims-')}.c{number}"] = raw
        for step, answer in responses.items():
            content = answer.get("content")
            if step.startswith("check-") and isinstance(content, dict):
                if content.get("supported") is not True:
                    raw = claims[step.removeprefix("check-")]
                    name = f"{path.parent.name}/{step}"
                    flags.append((name, raw["text"], raw["passage"], content["reason"]))
    return flags


RECORDED_EXPECTED = {
    "open-models-week/check-s1.c3": "The article says 71; the draft said 95.",
    # Both say 15: the number is there, only the month differs, so no number sentence.
    "open-models-week/check-s2.c2": (
        "The article does not say this. The checking model's reason: \"The passage gives "
        '15 December."'
    ),
    # "ten times" and "twice" are words, not figures.
    "open-models-week/check-s3.c2": (
        "The article does not say this. The checking model's reason: \"The passage says "
        'twice, not ten times."'
    ),
    "science-weekly/check-s1.c3": (
        "The article does not say this. The checking model's reason: \"The passage says "
        'nothing about habitability."'
    ),
    "science-weekly/check-s2.c3": (
        "The article does not say this. The checking model's reason: \"The passage says it "
        'has not been tested."'
    ),
}


def test_every_recorded_flag_is_covered():
    assert sorted(step for step, *_ in _recorded_flags()) == sorted(RECORDED_EXPECTED)


@pytest.mark.parametrize("step, claim, passage, reason", _recorded_flags(), ids=lambda v: v)
def test_recorded_flags_give_the_expected_sentence(step, claim, passage, reason):
    assert explain(claim, passage, "unsupported", reason) == RECORDED_EXPECTED[step]


def test_no_number_sentence_when_every_number_of_the_claim_is_in_the_passage():
    sentence = explain(
        "The model has 3 billion parameters and scores 71.",
        "It has 3 billion parameters. It scores 71 on MMLU.",
        "unsupported",
        "The passage is about another model.",
    )
    assert not sentence.startswith("The article says")
    assert sentence.startswith("The article does not say this.")


def test_a_missing_number_with_no_number_near_the_same_words_gives_no_number_sentence():
    sentence = explain(
        "The bill costs $5 million.",
        "Lawmakers passed the bill on 12 May after a long debate.",
        "unsupported",
        "",
    )
    assert sentence == "The article does not say this."


def test_each_passage_number_answers_only_the_claim_number_nearest_in_words():
    # 3B (in "Tiny-3B") is missing too, but only 95 shares its words with 71.
    sentence = explain(*CHANGED_NUMBER[:4])
    assert "3B" not in sentence


def test_two_changed_numbers_give_two_sentences():
    assert explain(
        "Revenue rose 12% to $40 million in the quarter.",
        "Revenue rose 8% to $35 million in the quarter.",
        "unsupported",
        "Wrong figures.",
    ) == (
        "The article says 8%; the draft said 12%. "
        "The article says $35 million; the draft said $40 million."
    )


def test_an_unchecked_claim_says_so_without_the_error():
    assert (
        explain("The chip ships in March.", "The chip ships soon.", "unchecked", "HTTP 503")
        == "This claim could not be checked."
    )


def test_an_unchecked_claim_still_gets_the_number_sentence():
    assert (
        explain("Tiny-3B scores 95 on MMLU.", "It scores 71 on MMLU.", "unchecked", "timeout")
        == "The article says 71; the draft said 95."
    )


def test_a_reason_with_a_figure_in_neither_text_falls_back_to_the_plain_status():
    assert (
        explain(
            "The result proves the planet is habitable.",
            "The team published the result on Thursday.",
            "unsupported",
            "The passage is 2 sentences long and says nothing of 300 kelvin.",
        )
        == "The article does not say this."
    )


def test_an_empty_reason_gives_the_status_alone():
    sentence = explain("A claim.", "A passage.", "unsupported", "  ")
    assert sentence == "The article does not say this."


EVERY_PAIR = [CHANGED_NUMBER, ADDED_NUMBER, NO_NUMBERS] + [
    (claim, passage, "unsupported", reason, None) for _, claim, passage, reason in _recorded_flags()
]


@pytest.mark.parametrize("claim, passage, status, reason, _", EVERY_PAIR)
def test_every_number_of_every_explanation_is_in_the_claim_or_the_passage(
    claim, passage, status, reason, _
):
    for number in extract_facts(explain(claim, passage, status, reason)).numbers:
        assert number in claim or number in passage
