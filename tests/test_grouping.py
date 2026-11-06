import json
import math
from pathlib import Path

import httpx
import pytest

from newsletter.grouping import LeftOut, group_items, item_text
from newsletter.llm import ConfigError, LLMClient, LLMError, Settings
from newsletter.sources import Item

FIXTURE = Path(__file__).parent / "fixtures" / "grouping" / "three_copies.json"


def make_item(title, link=None, summary="", feed_url="https://feed.example.com/rss"):
    return Item(
        title=title,
        link=link or f"https://example.com/{title.lower().replace(' ', '-')}",
        summary=summary,
        published=None,
        feed_url=feed_url,
        feed_title="Example",
        rights=None,
    )


class FakeEmbedder:
    """Stands in for `LLMClient.embed`: one vector per text, looked up by the text's first
    line (the topic, or the item's title)."""

    def __init__(self, vectors, error=None):
        self.vectors = vectors
        self.error = error
        self.calls = []

    def embed(self, step, inputs):
        self.calls.append((step, list(inputs)))
        if self.error:
            raise self.error
        return [list(self.vectors[text.split("\n", 1)[0]]) for text in inputs]


def load_fixture():
    data = json.loads(FIXTURE.read_text())
    items, vectors = [], {data["topic"]: data["topic_embedding"]}
    for raw in data["items"]:
        vectors[raw["title"]] = raw.pop("embedding")
        items.append(Item(published=None, rights=None, **raw))
    return data["topic"], items, vectors


def test_three_copies_of_one_story_become_one_story_with_three_sources():
    topic, items, vectors = load_fixture()
    llm = FakeEmbedder(vectors)

    grouping = group_items(items, llm, topic=topic)

    assert grouping.error is None
    assert grouping.left_out == []
    assert len(grouping.stories) == 2
    copies = [item for item in items if "Falcon-9" in item.title]
    first, second = grouping.stories.values()
    assert first == copies  # all three sources, in input order
    assert [item.link for item in first] == [
        "https://news.example.com/acme-falcon-9",
        "https://blog.example.net/2026/11/falcon-9",
        "https://ml.example.io/posts/falcon-9",
    ]
    assert [item.title for item in second] == ["City council approves new bike lanes"]


def test_one_embedding_call_for_the_topic_and_every_item():
    topic, items, vectors = load_fixture()
    llm = FakeEmbedder(vectors)

    group_items(items, llm, topic=topic)

    assert len(llm.calls) == 1
    step, inputs = llm.calls[0]
    assert step == "group"
    assert inputs[0] == topic
    assert inputs[1:] == [item_text(item) for item in items]


def test_stories_rank_closest_to_the_topic_first():
    items = [make_item("Far"), make_item("Close"), make_item("Middle")]
    llm = FakeEmbedder(
        {"the topic": [1, 0], "Far": [0, 1], "Close": [0.95, 0.05], "Middle": [0.6, 0.4]}
    )

    grouping = group_items(items, llm, topic="the topic", threshold=0.99)

    assert [story[0].title for story in grouping.stories.values()] == ["Close", "Middle", "Far"]
    assert list(grouping.stories) == ["s1", "s2", "s3"]
    assert [score for score in grouping.scores.values()] == sorted(
        grouping.scores.values(), reverse=True
    )


def test_ties_keep_input_order():
    items = [make_item("Left"), make_item("Best"), make_item("Right")]
    # Left and Right are equally close to the topic, and far from each other.
    llm = FakeEmbedder(
        {"the topic": [1, 0, 0], "Left": [0.5, 1, 0], "Best": [1, 0, 0], "Right": [0.5, 0, 1]}
    )

    grouping = group_items(items, llm, topic="the topic")

    assert [story[0].title for story in grouping.stories.values()] == ["Best", "Left", "Right"]

    llm.vectors["Best"] = [0.5, 0, -1]  # now all three tie, so input order holds
    grouping = group_items(items, llm, topic="the topic", threshold=0.999)
    assert [story[0].title for story in grouping.stories.values()] == ["Left", "Best", "Right"]


def test_a_failed_embedding_call_is_reported_with_every_item_listed():
    items = [make_item("One"), make_item("Two"), make_item("Three")]
    llm = FakeEmbedder({}, error=LLMError("group", "HTTP 500: upstream down"))

    grouping = group_items(items, llm, topic="the topic")

    assert grouping.error == "group: HTTP 500: upstream down"
    assert not grouping.ok
    assert grouping.left_out == [
        LeftOut(item.title, item.link, "ungrouped: the embedding call failed") for item in items
    ]
    # Never skipped: each item is still its own story, in input order.
    assert [story for story in grouping.stories.values()] == [[item] for item in items]


def test_a_failed_call_through_the_real_client_is_in_its_report_too():
    def server(request):
        return httpx.Response(500, json={"error": "down"})

    llm = LLMClient(
        Settings(models={"embedding": "embed-small"}, max_retries=0),
        transport=httpx.MockTransport(server),
    )
    items = [make_item("One"), make_item("Two")]

    grouping = group_items(items, llm, topic="the topic")

    assert grouping.error.startswith("group: HTTP 500")
    assert [entry.title for entry in grouping.left_out] == ["One", "Two"]
    assert llm.report()["failed_steps"] == ["group"]


def test_a_missing_embedding_model_is_a_config_error_not_a_quiet_fallback():
    llm = LLMClient(Settings(models={}), transport=httpx.MockTransport(lambda r: None))

    with pytest.raises(ConfigError, match="NEWSLETTER_EMBEDDING_MODEL"):
        group_items([make_item("One")], llm, topic="the topic")


def test_an_item_with_no_text_is_dropped_and_listed():
    blank = make_item("", link="https://example.com/blank")
    items = [make_item("One"), blank]
    llm = FakeEmbedder({"the topic": [1, 0], "One": [1, 0]})

    grouping = group_items(items, llm, topic="the topic")

    assert [story[0].title for story in grouping.stories.values()] == ["One"]
    assert grouping.left_out == [
        LeftOut("", "https://example.com/blank", "dropped: no title or summary to compare")
    ]
    assert len(llm.calls[0][1]) == 2  # the topic and "One" only


@pytest.mark.parametrize("vector", [[0, 0], [math.nan, 1], [math.inf, 0]])
def test_an_item_whose_embedding_cannot_be_compared_stays_ungrouped_and_listed(vector):
    items = [make_item("One"), make_item("Odd"), make_item("Two")]
    llm = FakeEmbedder({"the topic": [1, 0], "One": [1, 0], "Odd": vector, "Two": [1, 0.01]})

    grouping = group_items(items, llm, topic="the topic")

    stories = [[item.title for item in story] for story in grouping.stories.values()]
    assert stories == [["One", "Two"], ["Odd"]]
    assert grouping.left_out == [
        LeftOut("Odd", items[1].link, "ungrouped: its embedding is empty or not a number")
    ]


def test_a_topic_embedding_that_cannot_be_compared_is_an_error():
    items = [make_item("One"), make_item("Two")]
    llm = FakeEmbedder({"the topic": [0, 0], "One": [1, 0], "Two": [0, 1]})

    grouping = group_items(items, llm, topic="the topic")

    assert grouping.error == "group: the topic's embedding is empty or not a number"
    assert [story for story in grouping.stories.values()] == [[item] for item in items]
    assert len(grouping.left_out) == 2


def test_embeddings_of_different_lengths_are_an_error():
    items = [make_item("One"), make_item("Two")]
    llm = FakeEmbedder({"the topic": [1, 0], "One": [1, 0], "Two": [1, 0, 0]})

    grouping = group_items(items, llm, topic="the topic")

    assert grouping.error == "group: the embeddings have different lengths"
    assert [entry.title for entry in grouping.left_out] == ["One", "Two"]


def test_the_wrong_number_of_embeddings_is_an_error():
    class ShortEmbedder:
        def embed(self, step, inputs):
            return [[1, 0]]

    grouping = group_items([make_item("One")], ShortEmbedder(), topic="t")

    assert grouping.error == "group: expected 2 embeddings, got 1"
    assert grouping.left_out == [
        LeftOut("One", make_item("One").link, "ungrouped: the embeddings could not be used")
    ]


def test_a_story_ranks_by_its_closest_source():
    items = [make_item("Solo"), make_item("Copy far"), make_item("Copy near")]
    # The copies group; ranked by its first item alone the story would come after Solo.
    llm = FakeEmbedder(
        {
            "t": [1, 0, 0],
            "Solo": [0.3, 0, 0.95],
            "Copy far": [0.1, 1, 0],
            "Copy near": [0.45, 0.9, 0],
        }
    )

    grouping = group_items(items, llm, topic="t", threshold=0.9)

    stories = [[item.title for item in story] for story in grouping.stories.values()]
    assert stories == [["Copy far", "Copy near"], ["Solo"]]


def test_no_items_makes_no_call():
    llm = FakeEmbedder({})

    grouping = group_items([], llm, topic="the topic")

    assert grouping.stories == {}
    assert grouping.ok
    assert llm.calls == []


def test_only_blank_items_makes_no_call_and_lists_them():
    llm = FakeEmbedder({})

    grouping = group_items([make_item("", link="https://example.com/x")], llm, topic="t")

    assert llm.calls == []
    assert grouping.stories == {}
    assert [entry.reason for entry in grouping.left_out] == [
        "dropped: no title or summary to compare"
    ]
    assert not grouping.ok


@pytest.mark.parametrize("topic", ["", "   "])
def test_a_blank_topic_is_refused(topic):
    with pytest.raises(ValueError, match="topic"):
        group_items([make_item("One")], FakeEmbedder({}), topic=topic)


@pytest.mark.parametrize("threshold", [0, -0.5, 1.5, math.nan])
def test_a_threshold_outside_zero_to_one_is_refused(threshold):
    with pytest.raises(ValueError, match="threshold"):
        group_items([make_item("One")], FakeEmbedder({}), topic="t", threshold=threshold)


def test_a_story_joins_the_closest_group_not_the_first():
    items = [make_item("A"), make_item("B"), make_item("C")]
    # C is near both A and B, closer to B.
    llm = FakeEmbedder(
        {"t": [1, 0, 0], "A": [1, 0, 0], "B": [0.8, 0.6, 0], "C": [0.9397, 0.342, 0]}
    )

    grouping = group_items(items, llm, topic="t", threshold=0.9)

    stories = [[item.title for item in story] for story in grouping.stories.values()]
    assert stories == [["A"], ["B", "C"]]


def test_item_text_is_capped():
    item = make_item("Title", summary="word " * 10_000)

    text = item_text(item)

    assert text.startswith("Title\n")
    assert len(text) <= 4000


def test_to_dict_lists_the_report_fields():
    llm = FakeEmbedder({}, error=LLMError("group", "boom"))
    item = make_item("One")

    grouping = group_items([item], llm, topic="t")

    assert grouping.to_dict() == {
        "error": "group: boom",
        "left_out": [
            {"title": "One", "link": item.link, "reason": "ungrouped: the embedding call failed"}
        ],
        "ok": False,
    }
