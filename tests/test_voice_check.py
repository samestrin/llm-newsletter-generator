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
    # Numbers inside a URL or a claim id are not counted again as numbers.
    assert facts.numbers == ("3", "2026-11-02", "71", "12.5%")
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
        ("It costs $1,000.", "It costs $1000.", "number 1,000 was dropped"),
        ("It shipped [s1.c1].", "It shipped.", "claim id s1.c1 was dropped"),
        ("It shipped [s1.c1].", "It shipped [s1.c2].", "claim id s1.c1 was dropped"),
    ],
)
def test_a_rewrite_that_alters_or_drops_a_fact_is_reported(before, after, problem):
    assert problem in compare_facts(before, after)


def test_a_fact_repeated_fewer_times_is_reported():
    problems = compare_facts("Up 5%, then 5% again.", "Up 5% twice.")

    assert problems == ["number 5% was dropped"]


def test_text_without_facts_has_none():
    assert extract_facts("Plain words only.") == Facts((), (), (), ())
