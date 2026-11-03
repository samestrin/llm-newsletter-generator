# Changelog

## Unreleased

- `newsletter issue` writes the summaries in the writer's voice when `YOURVOICE_API_KEY` is set. It builds a yourvoice.dev profile from `NEWSLETTER_VOICE_SAMPLES` and caches it, then rewrites each summary with the reasoning model. A rewrite that drops, changes or adds a link, quoted string, number or claim id is rejected. A rejected rewrite, a failure or a timeout keeps the neutral summary and is listed in the run report. `newsletter.render.build_report` takes the voice pass too, so the Markdown and HTML run reports list it as well.
- `python -m newsletter.evals` prints an eval table (faithfulness rate, flag rate, planted false claims flagged, reference coverage, cost per issue) from two committed past feeds, each with a reference issue and recorded model responses that include planted false claims; it exits 1 if a planted claim is not flagged and runs in CI. `--live` runs the same feeds against real models (costs credit); `-o evals/results.md` saves the table.
- The shipped prompt templates are now protected by snapshot tests, and the retired `introduction.md` and `closing.md` prompts are gone (`item.md` is now `summary.md` until epic 03 T6 replaces the legacy writer).
- `newsletter.archive.build_site(issues, site_dir)` renders approved issues to a static site: an index and one page per issue at `issues/YYYY-MM-DD/index.html`. All feed and model text is HTML-escaped, only http and https links are rendered, the run report shows problems by feed host and count only, and the site is built in a temp dir and swapped in only when complete.
- Rewrote `README.md` for the new package: install with `uv`, the `newsletter` CLI, and the `NEWSLETTER_*` environment settings; dropped the version badge.
- `newsletter issue FEED_URL...` drafts a plain issue: one fast-model summary and one link per feed item, credits each feed, and ends with a run report of failed feeds, dropped entries and model calls with their tokens and cost. It exits 1 when anything failed.
- The prompt templates moved from `prompts/` to `src/newsletter/prompts/` so they ship with the package.
