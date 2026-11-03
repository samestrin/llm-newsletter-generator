"""The eval: recorded past feeds with planted false claims, scored without a model."""

from newsletter.evals.cli import main
from newsletter.evals.replay import ReplayClient
from newsletter.evals.run import (
    CASES_DIR,
    Case,
    CaseResult,
    load_cases,
    render_table,
    run_cases,
)

__all__ = [
    "CASES_DIR",
    "Case",
    "CaseResult",
    "ReplayClient",
    "load_cases",
    "main",
    "render_table",
    "run_cases",
]
