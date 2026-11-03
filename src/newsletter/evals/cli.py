"""`python -m newsletter.evals`: print the eval table.

Replay (the default) needs no network or key and exits 1 when a planted false claim was
not flagged. `--live` runs the same feeds against the models in the environment, spends
the owner's model credit, and exits 0 unless settings are missing (2).
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from newsletter.evals.run import CASES_DIR, load_cases, render_table, run_cases
from newsletter.llm import ConfigError, Settings


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    llm_transport: httpx.BaseTransport | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m newsletter.evals",
        description="Score claim checking on recorded past feeds with planted false claims.",
    )
    parser.add_argument("--live", action="store_true", help="call the real models (costs money)")
    parser.add_argument("--cases", type=Path, default=CASES_DIR, help="folder of eval cases")
    parser.add_argument("-o", "--output", type=Path, help="write the table here as well")
    args = parser.parse_args(argv)

    live = None
    if args.live:
        try:
            settings = Settings.from_env(env)
        except ConfigError as error:
            print(f"newsletter-eval: {error}", file=sys.stderr)
            return 2
        missing = [t for t in ("fast", "reasoning") if not settings.models.get(t)]
        if missing:
            names = ", ".join(f"NEWSLETTER_{t.upper()}_MODEL" for t in missing)
            print(f"newsletter-eval: set {names}", file=sys.stderr)
            return 2
        live = {"settings": settings, "transport": llm_transport}

    results = run_cases(load_cases(args.cases), live=live)
    mode = "live" if live else "replay of recorded responses"
    report = f"# Eval ({mode})\n\n{render_table(results)}"
    if live:
        report += (
            "\nLive runs have no planted claims and so no ground truth: faithfulness is "
            "n/a, and the flag rate is only how often the check flagged.\n"
        )
    sys.stdout.write(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")

    missed = [(r, cid) for r in results for cid in r.unflagged_planted]
    for result, claim_id in missed:
        print(
            f"newsletter-eval: planted false claim {claim_id} NOT flagged in {result.name}",
            file=sys.stderr,
        )
    errors = [(r, e) for r in results for e in r.story_errors]
    for result, error in errors:
        print(f"newsletter-eval: {result.name}: story failed, {error}", file=sys.stderr)
    return 1 if missed or errors else 0
