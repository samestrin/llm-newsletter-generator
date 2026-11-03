# Changelog

## Unreleased

- `python -m newsletter.evals` prints an eval table (faithfulness rate, flag rate, planted false claims flagged, reference coverage, cost per issue) from two committed past feeds, each with a reference issue and recorded model responses that include planted false claims; it exits 1 if a planted claim is not flagged and runs in CI. `--live` runs the same feeds against real models (costs credit); `-o evals/results.md` saves the table.
- Rewrote `README.md` for the new package: install with `uv`, the `newsletter` CLI, and the `NEWSLETTER_*` environment settings; dropped the version badge.
- `newsletter issue FEED_URL...` drafts a plain issue: one fast-model summary and one link per feed item, credits each feed, and ends with a run report of failed feeds, dropped entries and model calls with their tokens and cost. It exits 1 when anything failed.
- The prompt templates moved from `prompts/` to `src/newsletter/prompts/` so they ship with the package.
