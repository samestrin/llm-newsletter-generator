import json

import httpx
import pytest

from newsletter.editor import (
    CLAIMS_SCHEMA,
    parse_claims,
    summarize_stories,
    summarize_story,
)
from newsletter.llm import LLMClient, Settings
from newsletter.sources import Item

FIRST = Item(
    title="Lab ships a small model",
    link="https://lab.example.org/small-model",
    summary=(
        "The lab released a 3B parameter model on Tuesday.\n"
        "It runs on a laptop and scores 71 on the benchmark."
    ),
    published=None,
    feed_url="https://lab.example.org/atom.xml",
    feed_title="Lab blog",
    rights=None,
)
SECOND = Item(
    title="Small model from the lab",
    link="https://news.example.com/lab-model",
    summary="Analysts said the model is licensed under Apache 2.0.",
    published=None,
    feed_url="https://news.example.com/feed.xml",
    feed_title="News",
    rights=None,
)


class FakeModel:
    """An OpenAI-compatible endpoint that answers each call with the next reply in `replies`.

    A reply that is a dict is sent as JSON; a string is sent as is; an int is an HTTP error.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "bad request"})
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )


def make_llm(model):
    settings = Settings.from_env(
        {"NEWSLETTER_BASE_URL": "http://model.test/v1", "NEWSLETTER_FAST_MODEL": "fast-1"}
    )
    return LLMClient(settings, retry_backoff=0, transport=httpx.MockTransport(model))


def claim(text, passage):
    return {"text": text, "passage": passage}


# The schema the fast model is asked to follow.


def test_schema_asks_for_claims_each_with_text_and_passage():
    assert CLAIMS_SCHEMA["required"] == ["claims"]
    item = CLAIMS_SCHEMA["properties"]["claims"]["items"]
    assert item["required"] == ["text", "passage"]
    assert item["properties"]["text"] == {"type": "string"}
    assert item["properties"]["passage"] == {"type": "string"}
    assert item["additionalProperties"] is False
    assert CLAIMS_SCHEMA["additionalProperties"] is False


def test_the_fast_model_is_asked_for_structured_output_with_the_schema():
    model = FakeModel({"claims": []})
    summarize_story("s1", [FIRST], make_llm(model))

    request = model.requests[0]
    assert request["model"] == "fast-1"
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == CLAIMS_SCHEMA
    prompt = request["messages"][-1]["content"]
    assert FIRST.summary in prompt
    assert FIRST.title in prompt


def test_parse_claims_reads_a_reply_that_matches_the_schema():
    reply = json.dumps({"claims": [claim("It is small.", "a 3B parameter model")]})
    assert parse_claims(reply) == [{"text": "It is small.", "passage": "a 3B parameter model"}]


def test_parse_claims_accepts_a_reply_wrapped_in_a_code_fence():
    reply = '```json\n{"claims": [{"text": "t", "passage": "p"}]}\n```'
    assert parse_claims(reply) == [{"text": "t", "passage": "p"}]


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "[]",
        '{"summary": "no claims key"}',
        '{"claims": "not a list"}',
    ],
)
def test_parse_claims_rejects_a_reply_that_does_not_match_the_schema(reply):
    with pytest.raises(ValueError):
        parse_claims(reply)


# Each claim must be tied to a verbatim passage of the story's cleaned source text.


def test_claims_with_verbatim_passages_are_kept_with_their_source_link():
    model = FakeModel(
        {
            "claims": [
                claim("The lab released a 3B model.", "released a 3B parameter model on Tuesday"),
                claim("It uses the Apache license.", "licensed under Apache 2.0"),
            ]
        }
    )
    summary = summarize_story("s1", [FIRST, SECOND], make_llm(model))

    assert summary.error is None
    assert summary.dropped == []
    assert [(c.id, c.text, c.passage, c.source_url) for c in summary.claims] == [
        (
            "s1.c1",
            "The lab released a 3B model.",
            "released a 3B parameter model on Tuesday",
            FIRST.link,
        ),
        ("s1.c2", "It uses the Apache license.", "licensed under Apache 2.0", SECOND.link),
    ]


def test_a_claim_with_an_invented_passage_is_dropped_and_listed():
    model = FakeModel(
        {
            "claims": [
                claim("It runs on a laptop.", "It runs on a laptop"),
                claim("It beat GPT-5.", "The model beat GPT-5 on every benchmark."),
            ]
        }
    )
    run = summarize_stories({"s1": [FIRST]}, make_llm(model))

    [summary] = run.summaries
    assert [c.text for c in summary.claims] == ["It runs on a laptop."]
    [dropped] = summary.dropped
    assert dropped.id == "s1.c2"
    assert dropped.story_id == "s1"
    assert dropped.text == "It beat GPT-5."
    assert dropped.passage == "The model beat GPT-5 on every benchmark."
    assert "not found" in dropped.reason

    report = run.report()
    assert report["dropped_claims"] == [
        {
            "id": "s1.c2",
            "story_id": "s1",
            "text": "It beat GPT-5.",
            "passage": "The model beat GPT-5 on every benchmark.",
            "reason": dropped.reason,
        }
    ]
    assert report["failed_stories"] == []


def test_a_passage_that_is_only_close_to_the_source_is_dropped():
    # Verbatim means verbatim: a changed number or a paraphrase is not the source.
    model = FakeModel(
        {
            "claims": [
                claim("It scores 72.", "scores 72 on the benchmark"),
                claim("It runs on a laptop.", "it runs on a laptop"),
            ]
        }
    )
    summary = summarize_story("s1", [FIRST], make_llm(model))

    assert summary.claims == []
    assert [d.id for d in summary.dropped] == ["s1.c1", "s1.c2"]


def test_a_passage_spanning_two_sources_is_dropped():
    joined = "on the benchmark. Analysts said"
    model = FakeModel({"claims": [claim("Analysts liked it.", joined)]})
    summary = summarize_story("s1", [FIRST, SECOND], make_llm(model))

    assert summary.claims == []
    assert [d.id for d in summary.dropped] == ["s1.c1"]


def test_surrounding_whitespace_on_a_passage_is_ignored():
    model = FakeModel({"claims": [claim("It runs on a laptop.", "  It runs on a laptop\n")]})
    summary = summarize_story("s1", [FIRST], make_llm(model))

    assert [c.passage for c in summary.claims] == ["It runs on a laptop"]


@pytest.mark.parametrize(
    "bad",
    [
        {"text": "No passage.", "passage": ""},
        {"text": "Blank passage.", "passage": "   "},
        {"text": "", "passage": "It runs on a laptop"},
        {"text": "Passage is not text.", "passage": 3},
        {"text": "Missing passage."},
        "not an object",
    ],
)
def test_a_malformed_claim_is_dropped_and_listed(bad):
    model = FakeModel({"claims": [bad, claim("It runs on a laptop.", "It runs on a laptop")]})
    summary = summarize_story("s1", [FIRST], make_llm(model))

    assert [c.id for c in summary.claims] == ["s1.c2"]
    [dropped] = summary.dropped
    assert dropped.id == "s1.c1"
    assert dropped.reason


# A summary call that fails or returns something unusable is an error, never an empty success.


def test_a_failed_call_is_an_error_on_the_story_and_in_the_report():
    model = FakeModel(400, {"claims": [claim("It runs on a laptop.", "It runs on a laptop")]})
    run = summarize_stories({"s1": [FIRST], "s2": [SECOND, FIRST]}, make_llm(model))

    first, second = run.summaries
    assert first.claims == []
    assert "HTTP 400" in first.error
    assert second.error is None
    assert [c.id for c in second.claims] == ["s2.c1"]
    assert second.claims[0].source_url == FIRST.link
    assert run.report()["failed_stories"] == [{"story_id": "s1", "error": first.error}]


def test_an_unparsable_reply_is_an_error_on_the_story():
    model = FakeModel("Here is a summary: the lab made a model.")
    run = summarize_stories({"s1": [FIRST]}, make_llm(model))

    [summary] = run.summaries
    assert summary.claims == []
    assert summary.error.startswith("unparsable reply")
    assert run.report()["failed_stories"] == [{"story_id": "s1", "error": summary.error}]


def test_a_story_with_no_source_text_makes_no_call_and_has_no_claims():
    empty = Item("Title only", "https://e.example/x", "  ", None, "https://e.example/f", "E", None)
    model = FakeModel()
    summary = summarize_story("s1", [empty], make_llm(model))

    assert model.requests == []
    assert summary.claims == []
    assert summary.error is None


def test_each_story_is_its_own_fast_step():
    model = FakeModel({"claims": []}, {"claims": []})
    llm = make_llm(model)
    summarize_stories({"s1": [FIRST], "s2": [SECOND]}, llm)

    assert [(s["step"], s["tier"]) for s in llm.report()["steps"]] == [
        ("claims-s1", "fast"),
        ("claims-s2", "fast"),
    ]
