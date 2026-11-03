import copy
import json

import httpx
import pytest

from newsletter.editor import (
    EDIT_SCHEMA,
    CheckedClaim,
    CheckedStory,
    CheckRun,
    Claim,
    InvalidEdit,
    Prose,
    Theme,
    edit_issue,
    parse_edit,
)
from newsletter.llm import ConfigError, LLMClient, Settings


def claim(claim_id, text, status="supported"):
    passage = f"Passage for {text}"
    return CheckedClaim(
        Claim(claim_id, text, passage, f"https://example.org/{claim_id}"), status, "r"
    )


S1_TRUE = claim("s1.c1", "The lab released a 3B parameter model.")
S1_FALSE = claim("s1.c2", "The model scores 95 on the benchmark.", "unsupported")
S2_TRUE = claim("s2.c1", "The model is licensed under Apache 2.0.")
S2_UNCHECKED = claim("s2.c2", "The license changed last week.", "unchecked")
S3_FLAGGED = claim("s3.c1", "A rival lab will ship next month.", "unsupported")


def check_run():
    """Three stories: s1 and s2 have a supported claim each, s3 has only a flagged one."""
    return CheckRun(
        [
            CheckedStory("s1", [S1_TRUE, S1_FALSE]),
            CheckedStory("s2", [S2_TRUE, S2_UNCHECKED]),
            CheckedStory("s3", [S3_FLAGGED]),
        ]
    )


VALID = {
    "lead": "s2",
    "order": ["s2", "s1"],
    "themes": [
        {"name": "Open licenses", "claim_ids": ["s2.c1"]},
        {"name": "Small models", "claim_ids": ["s1.c1"]},
    ],
    "intro": {"text": "A small open model this week.", "claim_ids": ["s1.c1", "s2.c1"]},
    "why_it_matters": [
        {"story_id": "s2", "text": "Anyone can build on it.", "claim_ids": ["s2.c1"]},
        {"story_id": "s1", "text": "It fits on a laptop.", "claim_ids": ["s1.c1"]},
    ],
}


def edit(**changes):
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


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
        {
            "NEWSLETTER_BASE_URL": "http://model.test/v1",
            "NEWSLETTER_FAST_MODEL": "fast-1",
            "NEWSLETTER_REASONING_MODEL": "reason-1",
            "NEWSLETTER_MAX_RETRIES": "0",
        }
    )
    return LLMClient(settings, retry_backoff=0, transport=httpx.MockTransport(model))


def cited_ids(edition):
    prose = [edition.intro, *edition.themes, *edition.why_it_matters.values()]
    return {claim_id for part in prose for claim_id in part.claim_ids}


# The schema the reasoning call is asked to follow.


def test_schema_asks_for_lead_order_themes_intro_and_why_it_matters():
    assert EDIT_SCHEMA["required"] == ["lead", "order", "themes", "intro", "why_it_matters"]
    assert EDIT_SCHEMA["additionalProperties"] is False
    props = EDIT_SCHEMA["properties"]
    assert props["themes"]["items"]["required"] == ["name", "claim_ids"]
    assert props["intro"]["required"] == ["text", "claim_ids"]
    assert props["why_it_matters"]["items"]["required"] == ["story_id", "text", "claim_ids"]


def test_the_reasoning_call_gets_only_supported_claims_with_their_ids():
    model = FakeModel(VALID)
    edit_issue(check_run(), make_llm(model))

    request = model.requests[0]
    assert request["model"] == "reason-1"
    response_format = request["response_format"]
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == EDIT_SCHEMA
    prompt = request["messages"][-1]["content"]
    for checked in (S1_TRUE, S2_TRUE):
        assert checked.id in prompt
        assert checked.claim.text in prompt
    for checked in (S1_FALSE, S2_UNCHECKED, S3_FLAGGED):
        assert checked.id not in prompt
        assert checked.claim.text not in prompt
    assert "s3" not in prompt


# Valid output: the lead, order, themes, intro and why it matters, citing only checked claims.


def test_valid_output_is_used_and_cites_only_supported_claims():
    llm = make_llm(FakeModel(VALID))
    edition = edit_issue(check_run(), llm)

    assert edition.error is None
    assert edition.rejections == []
    assert edition.lead == "s2"
    assert edition.themes == [
        Theme("Open licenses", ["s2.c1"]),
        Theme("Small models", ["s1.c1"]),
    ]
    assert edition.intro == Prose("A small open model this week.", ["s1.c1", "s2.c1"])
    assert edition.why_it_matters == {
        "s2": Prose("Anyone can build on it.", ["s2.c1"]),
        "s1": Prose("It fits on a laptop.", ["s1.c1"]),
    }
    assert cited_ids(edition) <= {"s1.c1", "s2.c1"}
    assert [(s["step"], s["tier"]) for s in llm.report()["steps"]] == [("edit", "reasoning")]


def test_stories_are_in_the_editors_order_and_a_story_with_no_supported_claim_follows():
    edition = edit_issue(check_run(), make_llm(FakeModel(VALID)))

    assert [s.story_id for s in edition.stories] == ["s2", "s1", "s3"]
    # Flagged claims stay on their story, so they can still be shown as flagged.
    assert edition.stories[1].claims == [S1_TRUE, S1_FALSE]


def test_the_report_of_a_clean_edit_has_no_error_and_no_rejections():
    edition = edit_issue(check_run(), make_llm(FakeModel(VALID)))

    assert edition.report() == {"editor_error": None, "editor_rejections": []}


# The validator.


def test_parse_edit_accepts_valid_output_in_a_code_fence():
    draft = parse_edit(f"```json\n{json.dumps(VALID)}\n```", check_run().stories)
    assert draft.lead == "s2"


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (edit(intro={"text": "x", "claim_ids": ["s9.c1"]}), "unknown claim id 's9.c1'"),
        (edit(intro={"text": "x", "claim_ids": ["s1.c2"]}), "claim 's1.c2' is unsupported"),
        (
            edit(themes=[{"name": "a", "claim_ids": ["s2.c2"]}, VALID["themes"][1]]),
            "claim 's2.c2' is unchecked",
        ),
        (
            edit(
                why_it_matters=[
                    {"story_id": "s2", "text": "x", "claim_ids": ["s3.c1"]},
                    VALID["why_it_matters"][1],
                ]
            ),
            "claim 's3.c1' is unsupported",
        ),
        (edit(intro={"text": "x", "claim_ids": []}), "intro cites no claim"),
        (edit(intro={"text": "  ", "claim_ids": ["s1.c1"]}), "intro is empty"),
        (edit(intro={"text": "x", "claim_ids": [7]}), "intro claim ids are not strings"),
        (edit(themes=VALID["themes"][:1]), "2 or 3 themes"),
        (edit(themes=VALID["themes"] * 2), "2 or 3 themes"),
        (edit(themes=[{"name": "", "claim_ids": ["s1.c1"]}] * 2), "theme 1 is empty"),
        (edit(lead="s3"), "lead 's3' is not first in the order"),
        (edit(lead="s1"), "lead 's1' is not first in the order"),
        (edit(order=["s2"]), "order leaves out 's1'"),
        (edit(order=["s2", "s1", "s1"]), "order lists 's1' twice"),
        (edit(order=["s2", "s1", "s3"]), "order has unknown story 's3'"),
        (edit(why_it_matters=VALID["why_it_matters"][:1]), "no why it matters for 's1'"),
        (
            edit(
                why_it_matters=[
                    VALID["why_it_matters"][0],
                    {"story_id": "s1", "text": "x", "claim_ids": ["s2.c1"]},
                ]
            ),
            "why it matters for 's1' cites 's2.c1' of another story",
        ),
        (
            edit(why_it_matters=[*VALID["why_it_matters"], VALID["why_it_matters"][0]]),
            "why it matters for 's2' given twice",
        ),
        (
            edit(
                why_it_matters=[
                    *VALID["why_it_matters"],
                    {"story_id": "s9", "text": "x", "claim_ids": ["s1.c1"]},
                ]
            ),
            "why it matters for unknown story 's9'",
        ),
    ],
)
def test_parse_edit_rejects_invalid_output_and_says_why(data, problem):
    with pytest.raises(InvalidEdit) as caught:
        parse_edit(json.dumps(data), check_run().stories)
    assert any(problem in p for p in caught.value.problems), caught.value.problems


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        ("Here is the issue.", "not JSON"),
        ("[]", "reply is not an object"),
        ("null", "reply is not an object"),
        ("12", "reply is not an object"),
        (
            json.dumps({k: v for k, v in VALID.items() if k != "intro"}),
            "reply has no intro",
        ),
        (json.dumps(edit(lead=None)), "lead is not a story id"),
        (json.dumps(edit(order="s2,s1")), "order is not a list of story ids"),
        (json.dumps(edit(order=["s2", 1])), "order is not a list of story ids"),
        (json.dumps(edit(themes="Small models")), "themes and why it matters must be lists"),
        (json.dumps(edit(why_it_matters="s1")), "themes and why it matters must be lists"),
        (json.dumps(edit(intro="text")), "intro is not text with claim ids"),
        (
            json.dumps(edit(why_it_matters=["s1"])),
            "why it matters has an entry with no story id",
        ),
    ],
)
def test_parse_edit_rejects_a_reply_that_is_not_the_schema_and_says_why(reply, problem):
    with pytest.raises(InvalidEdit) as caught:
        parse_edit(reply, check_run().stories)
    assert any(problem in p for p in caught.value.problems), caught.value.problems


# Invalid output is retried once, then reported.


def test_output_citing_an_unknown_id_is_rejected_and_retried_once():
    bad = edit(intro={"text": "x", "claim_ids": ["s9.c1"]})
    model = FakeModel(bad, VALID)
    llm = make_llm(model)
    edition = edit_issue(check_run(), llm)

    assert edition.error is None
    assert edition.lead == "s2"
    assert len(edition.rejections) == 1
    assert "unknown claim id 's9.c1'" in edition.rejections[0]
    # The retry shows the model its rejected reply and why it was rejected.
    retry = model.requests[1]["messages"]
    assert retry[:-2] == model.requests[0]["messages"]
    assert retry[-2] == {"role": "assistant", "content": json.dumps(bad)}
    assert "unknown claim id 's9.c1'" in retry[-1]["content"]
    assert [s["step"] for s in llm.report()["steps"]] == ["edit", "edit-retry"]
    assert edition.report()["editor_rejections"] == edition.rejections


def test_output_citing_an_unsupported_id_twice_is_reported_as_an_error():
    bad = edit(intro={"text": "x", "claim_ids": ["s1.c2"]})
    model = FakeModel(bad, bad)
    edition = edit_issue(check_run(), make_llm(model))

    assert len(model.requests) == 2
    assert edition.error is not None
    assert "claim 's1.c2' is unsupported" in edition.error
    assert len(edition.rejections) == 2
    # Nothing from the rejected prose is used, and the stories are still listed in order.
    assert edition.lead is None
    assert edition.intro is None
    assert edition.themes == []
    assert edition.why_it_matters == {}
    assert [s.story_id for s in edition.stories] == ["s1", "s2", "s3"]
    assert edition.report() == {
        "editor_error": edition.error,
        "editor_rejections": edition.rejections,
    }


@pytest.mark.parametrize("reply", ["null", "12"])
def test_a_reply_that_is_json_but_not_an_object_is_retried_then_reported(reply):
    model = FakeModel(reply, reply)
    edition = edit_issue(check_run(), make_llm(model))

    assert len(model.requests) == 2
    assert "reply is not an object" in edition.error
    assert len(edition.rejections) == 2
    assert [s.story_id for s in edition.stories] == ["s1", "s2", "s3"]


def test_a_reply_that_is_json_but_not_an_object_is_retried_and_a_good_retry_is_used():
    edition = edit_issue(check_run(), make_llm(FakeModel("null", VALID)))

    assert edition.error is None
    assert edition.lead == "s2"
    assert edition.rejections == ["edit: reply is not an object"]


def test_an_unparsable_reply_is_retried_then_reported():
    edition = edit_issue(check_run(), make_llm(FakeModel("Sure!", "Sure!")))

    assert edition.error is not None
    assert "not JSON" in edition.error
    assert [s.story_id for s in edition.stories] == ["s1", "s2", "s3"]


# A failed reasoning call is an error in the run report, with the stories still in order.


def test_a_failing_reasoning_call_ends_in_a_reported_error_with_stories_in_order():
    model = FakeModel(500)
    llm = make_llm(model)
    edition = edit_issue(check_run(), llm)

    assert len(model.requests) == 1
    assert "HTTP 500" in edition.error
    assert edition.report()["editor_error"] == edition.error
    assert edition.lead is None
    assert [s.story_id for s in edition.stories] == ["s1", "s2", "s3"]
    assert llm.report()["failed_steps"] == ["edit"]


def test_a_failing_retry_call_ends_in_a_reported_error():
    bad = edit(intro={"text": "x", "claim_ids": ["s9.c1"]})
    llm = make_llm(FakeModel(bad, 500))
    edition = edit_issue(check_run(), llm)

    assert "HTTP 500" in edition.error
    assert len(edition.rejections) == 1
    assert edition.lead is None
    assert [s.story_id for s in edition.stories] == ["s1", "s2", "s3"]
    assert llm.report()["failed_steps"] == ["edit-retry"]


def test_with_no_supported_claim_there_is_no_call_and_an_error():
    model = FakeModel()
    run = CheckRun([CheckedStory("s1", [S1_FALSE]), CheckedStory("s2", error="HTTP 500")])
    edition = edit_issue(run, make_llm(model))

    assert model.requests == []
    assert edition.error == "no supported claims to edit from"
    assert [s.story_id for s in edition.stories] == ["s1", "s2"]


def test_a_missing_reasoning_model_is_a_config_error():
    settings = Settings.from_env({"NEWSLETTER_BASE_URL": "http://model.test/v1"})
    llm = LLMClient(settings, transport=httpx.MockTransport(FakeModel()))

    with pytest.raises(ConfigError, match="no reasoning model"):
        edit_issue(check_run(), llm)
