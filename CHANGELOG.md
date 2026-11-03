# Changelog

## Unreleased

- `newsletter issue FEED_URL...` drafts a plain issue: one fast-model summary and one link per feed item, credits each feed, and ends with a run report of failed feeds, dropped entries and model calls with their tokens and cost. It exits 1 when anything failed.
- The prompt templates moved from `prompts/` to `src/newsletter/prompts/` so they ship with the package.
