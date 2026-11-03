import json

import httpx
import pytest

from newsletter.editor import (
    CHECK_SCHEMA,
    Claim,
    ClaimsRun,
    DroppedClaim,
    StorySummary,
    check_claim,
    check_claims,
    parse_check,
)
from newsletter.llm import ConfigError, LLMClient, Settings

TRUE = Claim(
    id="s1.c1",
    text="The lab released a 3B parameter model.",
    passage="The lab released a 3B parameter model on Tuesday.",
    source_url="https://lab.example.org/small-model",
)
# Planted: the passage is real, but it does not say what the claim says.
FALSE = Claim(
    id="s1.c2",
    text="The model scores 95 on the benchmark.",
    passage="It runs on a laptop and scores 71 on the benchmark.",
    source_url="https://lab.example.org/small-model",
)
OTHER = Claim(
    id="s2.c1",
    text="The model is licensed under Apache 2.0.",
    passage="Analysts said the model is licensed under Apache 2.0.",
    source_url="https://news.example.com/lab-model",
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


def verdict(supported, reason="because"):
    return {"supported": supported, "reason": reason}


def claims_run(*summaries):
    return ClaimsRun(list(summaries))


# The schema the check call is asked to follow.


def test_schema_asks_for_a_yes_or_no_and_a_reason():
    assert CHECK_SCHEMA["required"] == ["supported", "reason"]
    assert CHECK_SCHEMA["properties"]["supported"] == {"type": "boolean"}
    assert CHECK_SCHEMA["properties"]["reason"] == {"type": "string"}
    assert CHECK_SCHEMA["additionalProperties"] is False


def test_the_check_call_gets_the_claim_and_its_passage_with_the_schema():
    model = FakeModel(verdict(True))
    check_claim(TRUE, make_llm(model))

    request = model.requests[0]
    assert request["model"] == "fast-1"
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == CHECK_SCHEMA
    prompt = request["messages"][-1]["content"]
    assert TRUE.text in prompt
    assert TRUE.passage in prompt


def test_parse_check_reads_a_reply_that_matches_the_schema():
    assert parse_check(json.dumps(verdict(False, "says 71"))) == (False, "says 71")


def test_parse_check_accepts_a_reply_wrapped_in_a_code_fence():
    assert parse_check('```json\n{"supported": true, "reason": "r"}\n```') == (True, "r")


@pytest.mark.parametrize(
    "reply",
    [
        "Yes, it is supported.",
        "[]",
        '{"reason": "no verdict"}',
        '{"supported": "true", "reason": "a string is not a yes"}',
        '{"supported": 1, "reason": "a number is not a yes"}',
        '{"supported": null, "reason": "r"}',
        '{"supported": true, "reason": 5}',
    ],
)
def test_parse_check_rejects_a_reply_that_does_not_match_the_schema(reply):
    with pytest.raises(ValueError):
        parse_check(reply)


# Each claim is checked against its passage.


def test_a_supported_claim_is_supported_and_not_flagged():
    checked = check_claim(TRUE, make_llm(FakeModel(verdict(True, "the passage says so"))))

    assert checked.status == "supported"
    assert checked.flagged is False
    assert checked.reason == "the passage says so"
    assert checked.claim == TRUE
    assert checked.id == "s1.c1"


def test_a_planted_false_claim_is_flagged():
    checked = check_claim(FALSE, make_llm(FakeModel(verdict(False, "the passage says 71"))))

    assert checked.status == "unsupported"
    assert checked.flagged is True
    assert checked.reason == "the passage says 71"


def test_a_failing_check_call_marks_the_claim_unchecked_and_flagged():
    checked = check_claim(TRUE, make_llm(FakeModel(400)))

    assert checked.status == "unchecked"
    assert checked.flagged is True
    assert "HTTP 400" in checked.reason


def test_an_unparsable_check_reply_marks_the_claim_unchecked_and_flagged():
    checked = check_claim(TRUE, make_llm(FakeModel("Looks right to me.")))

    assert checked.status == "unchecked"
    assert checked.flagged is True
    assert checked.reason.startswith("unparsable reply")


def test_a_reply_that_says_supported_as_a_string_is_never_supported():
    checked = check_claim(TRUE, make_llm(FakeModel({"supported": "yes", "reason": "r"})))

    assert checked.status == "unchecked"
    assert checked.flagged is True


# A whole run: flagged claims stay in the draft, in order, and are in the run report.


def test_flagged_claims_stay_in_the_draft_in_order_and_are_in_the_report():
    model = FakeModel(verdict(True), verdict(False, "the passage says 71"), 400)
    run = check_claims(
        claims_run(StorySummary("s1", [TRUE, FALSE]), StorySummary("s2", [OTHER])),
        make_llm(model),
    )

    first, second = run.stories
    assert [(c.id, c.status) for c in first.claims] == [
        ("s1.c1", "supported"),
        ("s1.c2", "unsupported"),
    ]
    assert [(c.id, c.status) for c in second.claims] == [("s2.c1", "unchecked")]
    assert [c.id for c in first.supported] == ["s1.c1"]
    assert second.supported == []

    report = run.report()
    assert report["flagged_claims"] == [
        {
            "id": "s1.c2",
            "story_id": "s1",
            "text": FALSE.text,
            "passage": FALSE.passage,
            "source_url": FALSE.source_url,
            "status": "unsupported",
            "reason": "the passage says 71",
        },
        {
            "id": "s2.c1",
            "story_id": "s2",
            "text": OTHER.text,
            "passage": OTHER.passage,
            "source_url": OTHER.source_url,
            "status": "unchecked",
            "reason": second.claims[0].reason,
        },
    ]
    assert report["failed_checks"] == [
        {"id": "s2.c1", "story_id": "s2", "error": second.claims[0].reason}
    ]


def test_a_failing_check_call_is_in_the_report_and_its_step_is_failed():
    llm = make_llm(FakeModel(400))
    run = check_claims(claims_run(StorySummary("s1", [TRUE])), llm)

    [checked] = run.stories[0].claims
    assert checked.status == "unchecked"
    assert run.report()["failed_checks"] == [
        {"id": "s1.c1", "story_id": "s1", "error": checked.reason}
    ]
    assert "HTTP 400" in checked.reason
    assert llm.report()["failed_steps"] == ["check-s1.c1"]


def test_each_claim_is_its_own_fast_step():
    llm = make_llm(FakeModel(verdict(True), verdict(True)))
    check_claims(claims_run(StorySummary("s1", [TRUE]), StorySummary("s2", [OTHER])), llm)

    assert [(s["step"], s["tier"]) for s in llm.report()["steps"]] == [
        ("check-s1.c1", "fast"),
        ("check-s2.c1", "fast"),
    ]


def test_a_story_keeps_its_dropped_claims_and_error_from_the_summary_step():
    dropped = DroppedClaim("s1.c3", "s1", "t", "p", "passage not found")
    model = FakeModel(verdict(True))
    run = check_claims(
        claims_run(StorySummary("s1", [TRUE], [dropped]), StorySummary("s2", error="HTTP 500")),
        make_llm(model),
    )

    first, second = run.stories
    assert first.dropped == [dropped]
    assert second.error == "HTTP 500"
    assert second.claims == []
    assert len(model.requests) == 1


def test_a_missing_fast_model_is_a_config_error_not_an_unparsable_reply():
    settings = Settings.from_env({"NEWSLETTER_BASE_URL": "http://model.test/v1"})
    llm = LLMClient(settings, transport=httpx.MockTransport(FakeModel()))

    with pytest.raises(ConfigError, match="no fast model"):
        check_claim(TRUE, llm)
