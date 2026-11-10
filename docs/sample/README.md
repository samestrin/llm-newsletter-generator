# Sample issue (Automated demo)

**Automated demo.** A model drafted this sample from public AI news feeds with no human
review. Every summary is marked "Flagged: unchecked": this plain issue checks no claim.

The sample was generated once, by the command below, and committed. CI never
regenerates it, and no model key is needed to read it.

- **Model:** `kimi-k3`
- **Date:** 2026-11-10
- **Feeds:**
  - https://openai.com/news/rss.xml
  - https://blog.google/technology/ai/rss/
  - https://huggingface.co/blog/feed.xml
  - https://www.technologyreview.com/topic/artificial-intelligence/feed
  - https://bair.berkeley.edu/blog/feed.xml

## Files

- [`issue.md`](issue.md): the issue as Markdown.
- [`email.html`](email.html): the issue as an HTML email.
- [`site/index.html`](site/index.html): the archive page, linking to the issue's page
  under `site/issues/<date>/`.

All three come from the same draft, and every page opens offline.

## Command

Run once from the repo root, through litellm. `NEWSLETTER_BASE_URL` and
`NEWSLETTER_API_KEY` are set in your own shell and never written here. `--date` is left
out, so the issue is dated the day it runs.

```sh
NEWSLETTER_FAST_MODEL=kimi-k3 NEWSLETTER_REASONING_MODEL=kimi-k3 \
uv run newsletter archive --demo \
  --title "AI News (automated demo)" \
  --topic "AI" \
  --per-feed 2 \
  --site docs/sample/site \
  --markdown docs/sample/issue.md \
  --html docs/sample/email.html \
  https://openai.com/news/rss.xml \
  https://blog.google/technology/ai/rss/ \
  https://huggingface.co/blog/feed.xml \
  https://www.technologyreview.com/topic/artificial-intelligence/feed \
  https://bair.berkeley.edu/blog/feed.xml
```

Then set **Date** above to the day it ran (`YYYY-MM-DD`) and commit `docs/sample/`.
`tests/test_sample.py` then checks that every file the command writes is here and
labeled, and that the pages load nothing from outside.

To use `glm-5.3` instead, name it in **Model** and in both variables of the command,
and give its litellm route high thinking effort (never low: it loops).
