"""The README's numbers and pictures come from the repo, not from someone's typing.

- `evals/results.md` is what `python -m newsletter.evals` prints now.
- The README's eval table is the table in `evals/results.md`.
- The issue the browser test screenshots is what the renderer makes now.
- Every image the README shows is committed.
"""

import json
import re
from pathlib import Path

import readme_issue
from newsletter.evals.cli import main as eval_main
from newsletter.evals.run import CASES_DIR

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "evals" / "results.md"


def table(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("|")]


def test_results_md_is_what_the_eval_prints(tmp_path):
    out = tmp_path / "results.md"
    assert eval_main(["-o", str(out)]) == 0
    assert RESULTS.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")


def test_results_md_says_it_is_a_replay():
    assert "replay of recorded responses" in RESULTS.read_text(encoding="utf-8").splitlines()[0]


def test_the_readme_eval_table_is_the_one_in_results_md():
    results = table(RESULTS.read_text(encoding="utf-8"))
    assert results
    readme = README.read_text(encoding="utf-8").splitlines()
    start = readme.index(results[0])
    assert readme[start : start + len(results)] == results


def test_the_screenshotted_issue_is_what_the_renderer_makes():
    for name, text in readme_issue.render().items():
        assert (readme_issue.FIXTURES / name).read_text(encoding="utf-8") == text, name


def test_the_screenshotted_issue_shows_its_planted_claims_flagged():
    fixtures = readme_issue.render()
    assert "Flagged: unsupported" in fixtures["issue.html"]
    flagged = {flag["id"] for flag in json.loads(fixtures["draft.json"])["flags"]}
    planted = json.loads((CASES_DIR / readme_issue.CASE / "case.json").read_text())
    assert flagged == set(planted["planted_false"])


def test_every_readme_image_is_committed():
    images = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", README.read_text(encoding="utf-8"))
    local = [i for i in images if not i.startswith(("http://", "https://"))]
    assert {
        "docs/img/review-page-light.png",
        "docs/img/review-page-dark.png",
        "docs/img/review-page-phone.png",
        "docs/img/issue.png",
    } <= set(local)
    for image in local:
        assert (ROOT / image).is_file(), image


def test_every_readme_link_to_a_repo_file_exists():
    links = re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s#]+)", README.read_text(encoding="utf-8"))
    for link in links:
        if not link.startswith(("http://", "https://", "mailto:")):
            assert (ROOT / link).exists(), link
