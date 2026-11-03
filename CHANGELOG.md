# Changelog

## Unreleased

- `newsletter issue` writes the summaries in the writer's voice when `YOURVOICE_API_KEY` is set. It builds a yourvoice.dev profile from `NEWSLETTER_VOICE_SAMPLES` and caches it, then rewrites each summary with the reasoning model. A rewrite that drops, changes or adds a link, quoted string, number or claim id is rejected. A rejected rewrite, a failure or a timeout keeps the neutral summary and is listed in the run report.
- Rewrote `README.md` for the new package: install with `uv`, the `newsletter` CLI, and the `NEWSLETTER_*` environment settings; dropped the version badge.
- `newsletter issue FEED_URL...` drafts a plain issue: one fast-model summary and one link per feed item, credits each feed, and ends with a run report of failed feeds, dropped entries and model calls with their tokens and cost. It exits 1 when anything failed.
- The prompt templates moved from `prompts/` to `src/newsletter/prompts/` so they ship with the package.
