import pytest

from newsletter.voice import Facts, compare_facts, extract_facts

NEUTRAL = (
    "The lab released a 3B parameter model on 2026-11-02 [s1.c1]. It scores 71 on the "
    'benchmark, up 12.5% [s1.c2]. "We wanted it to run on a laptop," the lead said '
    "[s2.c1]. Read the [announcement](<https://lab.example.org/small-model?id=42>)."
)


def test_extracts_urls_quotes_numbers_and_claim_ids():
    facts = extract_facts(NEUTRAL)

    assert facts.urls == ("https://lab.example.org/small-model?id=42",)
    assert facts.quotes == ("We wanted it to run on a laptop,",)
    # Numbers inside a URL or a claim id are not counted again as numbers, and a unit
    # written onto a number is part of it.
    assert facts.numbers == ("3B", "2026-11-02", "71", "12.5%")
    assert facts.claim_ids == ("s1.c1", "s1.c2", "s2.c1")


def test_a_rewrite_that_keeps_every_fact_passes():
    rewrite = (
        "Big news: the lab put out a 3B parameter model on 2026-11-02 [s1.c1], and it lands "
        "a 71 on the benchmark, a 12.5% jump [s1.c2]. In the lead's words: “We wanted it to "
        "run on a laptop,” [s2.c1]. The [announcement](<https://lab.example.org/small-model?id=42>)"
        " has more."
    )

    assert compare_facts(NEUTRAL, rewrite) == []


@pytest.mark.parametrize(
    ("before", "after", "problem"),
    [
        (
            "See https://a.example/x.",
            "See https://a.example/y.",
            "link https://a.example/x was dropped",
        ),
        ("See it.", "See https://evil.example/.", "link https://evil.example/ was added"),
        ('He said "it is fast".', 'He said "it is quick".', 'quote "it is fast" was dropped'),
        ('He said "it is fast".', "He said it is fast.", 'quote "it is fast" was dropped'),
        ("It scores 71.", "It scores 95.", "number 71 was dropped"),
        ("It scores 71.", "It scores seventy-one.", "number 71 was dropped"),
        ("It costs $1,000.", "It costs $1000.", "number $1,000 was dropped"),
        ("It costs $1,000.", "It costs €1,000.", "number $1,000 was dropped"),
        ("It fell -5% last week.", "It rose 5% last week.", "number -5% was dropped"),
        ("It fell −5% last week.", "It rose 5% last week.", "number −5% was dropped"),
        ("It rose +5%.", "It rose 5%.", "number +5% was dropped"),
        ("The model is 3B params.", "The model is 3M params.", "number 3B was dropped"),
        ("Sales hit 5 million.", "Sales hit 5 billion.", "number 5 million was dropped"),
        ("Only .5 million.", "Only 5 million.", "number .5 million was dropped"),
        ("He said 'it is fast'.", "He said 'it is slow'.", 'quote "it is fast" was dropped'),
        ("He said ‘it’s fast’.", "He said ‘it’s slow’.", 'quote "it’s fast" was dropped'),
        ("It shipped [s1.c1].", "It shipped.", "claim id s1.c1 was dropped"),
        ("It shipped [s1.c1].", "It shipped [s1.c2].", "claim id s1.c1 was dropped"),
    ],
)
def test_a_rewrite_that_alters_or_drops_a_fact_is_reported(before, after, problem):
    assert problem in compare_facts(before, after)


def test_a_fact_repeated_fewer_times_is_reported():
    problems = compare_facts("Up 5%, then 5% again.", "Up 5% twice.")

    assert problems == ["number 5% was dropped"]


def test_apostrophes_are_not_quotes():
    facts = extract_facts("It's the lead's model, and the writers' pick. GPT-4 isn't one.")

    assert facts.quotes == ()
    # The hyphen in a name is not a minus sign.
    assert facts.numbers == ("4",)


def test_a_quote_is_read_whole_even_when_it_holds_a_link():
    text = 'The post said "see https://a.example/x for 3 notes".'

    assert extract_facts(text).quotes == ("see https://a.example/x for 3 notes",)
    assert compare_facts(text, 'The post said "see https://a.example/x for notes".') == [
        'quote "see https://a.example/x for 3 notes" was dropped',
        'quote "see https://a.example/x for notes" was added',
        "number 3 was dropped",
    ]


def test_text_without_facts_has_none():
    assert extract_facts("Plain words only.") == Facts((), (), (), ())
