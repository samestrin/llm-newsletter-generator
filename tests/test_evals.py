import json
import shutil
from pathlib import Path

import httpx
import pytest

from newsletter.evals import (
    CASES_DIR,
    ReplayClient,
    load_cases,
    main,
    render_table,
    run_cases,
)
from newsletter.llm import LLMError, Settings


def test_bundled_cases_flag_every_planted_false_claim():
    results = run_cases(load_cases())
    assert len(results) >= 2
    planted = sum(r.planted for r in results)
    assert planted >= 3
    assert sum(r.planted_flagged for r in results) == planted
    assert all(r.claims_dropped == 0 for r in results)
    assert all(r.story_errors == [] for r in results)
    assert [r.planted for r in results] == [len(c.planted_false) for c in load_cases()]
    assert all(r.faithfulness == 1.0 for r in results)
    assert all(0 < r.flag_rate < 1 for r in results)
    assert all(r.cost and r.cost > 0 for r in results)
    assert all(r.reference_coverage == 1.0 for r in results)


def test_planted_claim_recorded_as_supported_is_counted_against_faithfulness(tmp_path):
    cases = tmp_path / "cases"
    shutil.copytree(CASES_DIR, cases)
    case_dir = sorted(cases.iterdir())[0]
    recorded = json.loads((case_dir / "recorded.json").read_text())
    planted = json.loads((case_dir / "case.json").read_text())["planted_false"][0]
    recorded["responses"][f"check-{planted}"]["content"] = {"supported": True, "reason": "ok"}
    (case_dir / "recorded.json").write_text(json.dumps(recorded))

    [bad] = [r for r in run_cases(load_cases(cases)) if r.name == case_dir.name]
    assert bad.planted_flagged == bad.planted - 1
    assert bad.faithfulness < 1.0


def test_replay_client_fails_on_a_step_that_was_not_recorded():
    client = ReplayClient(Settings(models={"fast": "m"}), {})
    with pytest.raises(LLMError, match="not recorded"):
        client.complete("claims-s1", [{"role": "user", "content": "x"}])
    assert client.report()["failed_steps"] == ["claims-s1"]


def test_replay_client_reports_recorded_usage_and_cost():
    recorded = {
        "a": {"content": "hi", "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.5}}
    }
    client = ReplayClient(Settings(models={"fast": "m"}), recorded)
    assert client.complete("a", []) == "hi"
    assert client.report()["total_cost"] == 0.5
    assert client.records[0].prompt_tokens == 5


def test_table_has_a_row_per_case_and_a_total(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Faithfulness" in out and "Flag rate" in out and "Cost per issue" in out
    assert "Planted false claims flagged" in out
    for case in load_cases():
        assert case.name in out
    assert "| Total |" in out
    assert render_table(run_cases(load_cases())).startswith("| Case")


def test_main_exits_1_when_a_planted_claim_is_not_flagged(tmp_path, capsys):
    cases = tmp_path / "cases"
    shutil.copytree(CASES_DIR, cases)
    case_dir = sorted(cases.iterdir())[0]
    recorded = json.loads((case_dir / "recorded.json").read_text())
    planted = json.loads((case_dir / "case.json").read_text())["planted_false"][0]
    recorded["responses"][f"check-{planted}"]["content"] = {"supported": True, "reason": "ok"}
    (case_dir / "recorded.json").write_text(json.dumps(recorded))
    assert main(["--cases", str(cases)]) == 1
    assert "NOT flagged" in capsys.readouterr().err


def test_main_writes_the_table_to_a_file(tmp_path):
    out = tmp_path / "results.md"
    assert main(["-o", str(out)]) == 0
    assert out.read_text().startswith("# Eval")


def test_live_mode_needs_models(capsys):
    assert main(["--live"], env={}) == 2
    assert "NEWSLETTER_FAST_MODEL" in capsys.readouterr().err


def test_live_mode_runs_the_feeds_against_a_model(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # A model that gives no claims: nothing to check, nothing to edit.
        content = json.dumps({"claims": []})
        assert body["model"] in {"f", "r"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    env = {
        "NEWSLETTER_FAST_MODEL": "f",
        "NEWSLETTER_REASONING_MODEL": "r",
        "NEWSLETTER_FAST_PRICE_INPUT": "1",
        "NEWSLETTER_FAST_PRICE_OUTPUT": "1",
    }
    assert main(["--live"], env=env, llm_transport=httpx.MockTransport(handler)) == 0
    out = capsys.readouterr().out
    assert "live" in out
    # No labels live: faithfulness is not computed, never reported as a self-graded 100%.
    row = next(line for line in out.splitlines() if line.startswith("| open-models-week"))
    assert row.split("|")[2].strip() == "n/a"
    assert "no ground truth" in out


def test_a_step_missing_from_the_recording_fails_the_eval(tmp_path, capsys):
    cases = tmp_path / "cases"
    shutil.copytree(CASES_DIR, cases)
    case_dir = sorted(cases.iterdir())[0]
    recorded = json.loads((case_dir / "recorded.json").read_text())
    del recorded["responses"]["claims-s1"]
    (case_dir / "recorded.json").write_text(json.dumps(recorded))
    [bad, _] = run_cases(load_cases(cases))
    assert bad.story_errors and "claims-s1" in bad.story_errors[0]
    assert main(["--cases", str(cases)]) == 1
    assert "claims-s1" in capsys.readouterr().err


def test_output_creates_missing_folders(tmp_path):
    out = tmp_path / "evals" / "results.md"
    assert main(["-o", str(out)]) == 0
    assert out.is_file()


def test_load_cases_rejects_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cases(Path(tmp_path))
