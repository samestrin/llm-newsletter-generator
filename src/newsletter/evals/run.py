"""The eval: past feeds run through the claims, check and edit steps, and scored.

Each case is a folder: the feed as it was (`feed.xml`), the model's answers as recorded
(`recorded.json`), the issue an editor would run (`reference.md`) and `case.json`, which
names the claims that were planted false. A planted claim is a true passage with a false
claim written over it. Whether it is flagged is judged against that list, not by the
checker's own opinion, so the faithfulness rate is not self-graded.

Per case:

- faithfulness: of the claims the check passed, the share that are not planted false
- flag rate: flagged claims over all claims checked
- planted flagged: planted false claims the check flagged, over all planted
- cost: the issue's total model cost
- reference coverage: the reference issue's links the edition's supported claims cite

The replay needs no network and no key. Live mode runs the same feeds against the models
in the environment; it has no planted claims, so its faithfulness is the check's own share.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from newsletter.editor import Edition, check_claims, edit_issue, summarize_stories
from newsletter.evals.replay import ReplayClient
from newsletter.llm import LLMClient, Settings
from newsletter.sources.feed import parse_feed

CASES_DIR = Path(__file__).parent / "cases"

_LINK = re.compile(r"\]\(<?(https?://[^)>\s]+)")


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    topic: str
    feed: bytes
    feed_url: str
    planted_false: frozenset[str]
    reference: str
    recorded: dict[str, dict[str, Any]]
    models: dict[str, str]


@dataclass
class CaseResult:
    name: str
    claims: int = 0
    claims_dropped: int = 0
    flagged: int = 0
    supported: int = 0
    planted: int = 0
    planted_flagged: int = 0
    planted_passed: int = 0
    cost: float | None = None
    reference_links: int = 0
    reference_found: int = 0
    edit_error: str | None = None
    edit_rejections: int = 0
    unflagged_planted: list[str] = field(default_factory=list)

    @property
    def faithfulness(self) -> float | None:
        """Share of passed claims that are not planted false; None when none passed."""
        if not self.supported:
            return None
        return 1 - self.planted_passed / self.supported

    @property
    def flag_rate(self) -> float | None:
        return self.flagged / self.claims if self.claims else None

    @property
    def reference_coverage(self) -> float | None:
        return self.reference_found / self.reference_links if self.reference_links else None


def load_cases(directory: Path = CASES_DIR) -> list[Case]:
    import json

    folders = sorted(p for p in directory.iterdir() if (p / "case.json").is_file())
    if not folders:
        raise FileNotFoundError(f"no eval cases in {directory}")
    cases = []
    for folder in folders:
        meta = json.loads((folder / "case.json").read_text(encoding="utf-8"))
        recorded = json.loads((folder / "recorded.json").read_text(encoding="utf-8"))
        cases.append(
            Case(
                name=folder.name,
                title=meta["title"],
                topic=meta["topic"],
                feed=(folder / "feed.xml").read_bytes(),
                feed_url=meta["feed_url"],
                planted_false=frozenset(meta["planted_false"]),
                reference=(folder / "reference.md").read_text(encoding="utf-8"),
                recorded=recorded["responses"],
                models=recorded["models"],
            )
        )
    return cases


def run_cases(cases: list[Case], *, live: Mapping[str, Any] | None = None) -> list[CaseResult]:
    """Replay every case, or with `live` (`settings`, optional `transport`) call real models."""
    return [_run_case(case, live) for case in cases]


def _run_case(case: Case, live: Mapping[str, Any] | None) -> CaseResult:
    if live is None:
        settings = Settings(models=case.models)
        llm: LLMClient = ReplayClient(settings, case.recorded)
        planted = case.planted_false
    else:
        llm = LLMClient(live["settings"], transport=live.get("transport"))
        planted = frozenset()
    try:
        items = parse_feed(case.feed, case.feed_url).items
        stories = {f"s{n}": [item] for n, item in enumerate(items, start=1)}
        links = {sid: group[0].link for sid, group in stories.items()}
        claims = summarize_stories(stories, llm)
        checked = check_claims(claims, llm)
        edition = edit_issue(checked, llm)
        report = llm.report()
    finally:
        llm.close()

    result = CaseResult(case.name)
    result.claims_dropped = sum(len(s.dropped) for s in claims.summaries)
    for story in checked.stories:
        for claim in story.claims:
            result.claims += 1
            result.flagged += claim.flagged
            result.supported += not claim.flagged
            if claim.id in planted:
                result.planted += 1
                if claim.flagged:
                    result.planted_flagged += 1
                else:
                    result.planted_passed += 1
                    result.unflagged_planted.append(claim.id)
    result.cost = report["total_cost"] if report["cost_complete"] else None
    result.edit_error = edition.error
    result.edit_rejections = len(edition.rejections)
    _score_reference(result, case.reference, edition, links)
    return result


def _score_reference(
    result: CaseResult, reference: str, edition: Edition, links: Mapping[str, str]
) -> None:
    wanted = list(dict.fromkeys(_LINK.findall(reference)))
    cited = (
        {claim.claim.source_url for story in edition.stories for claim in story.supported}
        if edition.intro
        else set()
    )
    result.reference_links = len(wanted)
    result.reference_found = sum(1 for link in wanted if link in cited)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _cost(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def render_table(results: list[CaseResult]) -> str:
    """The results as a Markdown table, one row per case and a total."""
    lines = [
        "| Case | Faithfulness | Flag rate | Planted false claims flagged "
        "| Reference coverage | Cost per issue |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(_row(r.name, r))
    lines.append(_row("Total", _total(results), cost=_mean_cost(results)))
    return "\n".join(lines) + "\n"


def _row(name: str, r: CaseResult, cost: float | None | str = "") -> str:
    planted = f"{r.planted_flagged}/{r.planted}" if r.planted else "n/a"
    return (
        f"| {name} | {_pct(r.faithfulness)} | {_pct(r.flag_rate)} | {planted} "
        f"| {_pct(r.reference_coverage)} | {_cost(r.cost if cost == '' else cost)} |"
    )


def _total(results: list[CaseResult]) -> CaseResult:
    total = CaseResult("Total")
    for field_name in (
        "claims",
        "claims_dropped",
        "flagged",
        "supported",
        "planted",
        "planted_flagged",
        "planted_passed",
        "reference_links",
        "reference_found",
    ):
        setattr(total, field_name, sum(getattr(r, field_name) for r in results))
    return total


def _mean_cost(results: list[CaseResult]) -> float | None:
    if not results or any(r.cost is None for r in results):
        return None
    return sum(r.cost for r in results) / len(results)  # type: ignore[misc]
