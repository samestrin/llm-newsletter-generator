"""Group items that tell the same story, and rank the stories by the issue's topic.

One call to the `embedding` model embeds the topic and every item. An item joins the
story whose first item it is closest to, if that is at least `threshold` (cosine);
otherwise it starts a story of its own. Stories rank by their closest item to the topic,
and ties keep input order.

Nothing is skipped quietly. An item with no text to compare is dropped and listed. An
item whose embedding cannot be compared is its own story, ranked last and listed. When
the call fails or its result cannot be used, every item is its own story in input
order, each is listed, and the error is set. A missing `embedding` model raises
`ConfigError` like any other model call.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from newsletter.llm import LLMError
from newsletter.sources import Item

STEP = "group"
DEFAULT_THRESHOLD = 0.85
TEXT_LIMIT = 4000
"""Characters of an item sent to the model: the title, then the start of the summary."""


class Embedder(Protocol):
    def embed(self, step: str, inputs: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class LeftOut:
    """One line of the run report: an item that was dropped or not grouped."""

    title: str
    link: str
    reason: str


@dataclass
class Grouping:
    stories: dict[str, list[Item]] = field(default_factory=dict)
    """Story id to its items (its sources), closest to the topic first."""
    scores: dict[str, float | None] = field(default_factory=dict)
    """Story id to its closeness to the topic; None when it could not be compared."""
    left_out: list[LeftOut] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.left_out

    def to_dict(self) -> dict[str, Any]:
        """The run report's part: the error and every item left out."""
        return {
            "error": self.error,
            "left_out": [asdict(entry) for entry in self.left_out],
            "ok": self.ok,
        }


class _Unusable(Exception):
    pass


def item_text(item: Item) -> str:
    """What the model compares: the title, then the summary, capped."""
    return f"{item.title.strip()}\n{item.summary.strip()}"[:TEXT_LIMIT]


def group_items(
    items: Sequence[Item],
    llm: Embedder,
    *,
    topic: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> Grouping:
    if not topic.strip():
        raise ValueError("topic must not be blank")
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")

    grouping = Grouping()
    kept: list[Item] = []
    for item in items:
        if item.title.strip() or item.summary.strip():
            kept.append(item)
        else:
            grouping.left_out.append(
                LeftOut(item.title, item.link, "dropped: no title or summary to compare")
            )
    if not kept:
        return grouping

    try:
        vectors = llm.embed(STEP, [topic, *(item_text(item) for item in kept)])
        topic_vector, *item_vectors = _unit_vectors(vectors, len(kept) + 1)
        if topic_vector is None:
            raise _Unusable("the topic's embedding is empty or not a number")
    except LLMError as exc:
        return _ungrouped(grouping, kept, str(exc), "the embedding call failed")
    except _Unusable as exc:
        return _ungrouped(grouping, kept, f"{STEP}: {exc}", "the embeddings could not be used")

    # Each cluster: (unit vector of its first item, its items, its best topic score).
    clusters: list[tuple[list[float] | None, list[Item], float | None]] = []
    for item, vector in zip(kept, item_vectors, strict=True):
        if vector is None:
            grouping.left_out.append(
                LeftOut(item.title, item.link, "ungrouped: its embedding is empty or not a number")
            )
            clusters.append((None, [item], None))
            continue
        score = _dot(vector, topic_vector)
        best: tuple[int, float] | None = None
        for index, (first, _, _) in enumerate(clusters):
            if first is None:
                continue
            similarity = _dot(vector, first)
            if similarity >= threshold and (best is None or similarity > best[1]):
                best = (index, similarity)
        if best is None:
            clusters.append((vector, [item], score))
        else:
            first, members, top = clusters[best[0]]
            members.append(item)
            clusters[best[0]] = (first, members, max(top, score))

    # sorted() is stable, so ties keep the order the stories first appeared in.
    ranked = sorted(clusters, key=lambda c: -math.inf if c[2] is None else c[2], reverse=True)
    for number, (_, members, score) in enumerate(ranked, 1):
        grouping.stories[f"s{number}"] = members
        grouping.scores[f"s{number}"] = score
    return grouping


def _ungrouped(grouping: Grouping, items: list[Item], error: str, reason: str) -> Grouping:
    grouping.error = error
    for number, item in enumerate(items, 1):
        grouping.stories[f"s{number}"] = [item]
        grouping.scores[f"s{number}"] = None
        grouping.left_out.append(LeftOut(item.title, item.link, f"ungrouped: {reason}"))
    return grouping


def _unit_vectors(vectors: list[list[float]], count: int) -> list[list[float] | None]:
    """Each vector scaled to length 1, or None if it is empty, zero or not finite."""
    if len(vectors) != count:
        raise _Unusable(f"expected {count} embeddings, got {len(vectors)}")
    if len({len(vector) for vector in vectors}) > 1:
        raise _Unusable("the embeddings have different lengths")
    units: list[list[float] | None] = []
    for vector in vectors:
        norm = math.sqrt(math.fsum(x * x for x in vector)) if vector else 0.0
        if not math.isfinite(norm) or norm == 0:
            units.append(None)
        else:
            units.append([x / norm for x in vector])
    return units


def _dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))
