# llm-newsletter-generator

[![Star on GitHub](https://img.shields.io/github/stars/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/stargazers) [![Fork on GitHub](https://img.shields.io/github/forks/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/network/members) [![Watch on GitHub](https://img.shields.io/github/watchers/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/watchers)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Built with Python](https://img.shields.io/badge/Built%20with-Python-green)](https://www.python.org/)

**llm-newsletter-generator** is a Python package that generates text-only newsletters from RSS feeds using LLMs. See [VISION.md](VISION.md) for the plan.

The rebuild is in progress: `newsletter issue` drafts a plain issue from RSS and Atom feeds, in the writer's voice when a yourvoice.dev key is set; editing and grouping arrive with later work.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/samestrin/llm-newsletter-generator/
cd llm-newsletter-generator
uv sync
```

Dependencies are declared in `pyproject.toml`, which is also where the version lives.

## Usage

```bash
uv run newsletter --help
uv run newsletter --version
NEWSLETTER_FAST_MODEL=gpt-4o-mini NEWSLETTER_API_KEY=... \
  uv run newsletter issue https://example.com/feed.xml --title "AI Weekly" --topic AI -o issue.md
```

`newsletter issue` reads RSS or Atom feeds, summarizes each item with the fast model and writes the issue as Markdown with a link per item, then a run report listing any feed, entry or model call that failed and what each step cost. It exits 1 when anything failed, so a partial issue is never reported as a success. Use `--cache-dir` to keep fetched feeds between runs and `--date` to set the issue date.

## Configuration

The CLI talks to any OpenAI-compatible endpoint; all model settings come from the environment:

| Variable | Purpose |
| --- | --- |
| `NEWSLETTER_BASE_URL` | Endpoint, default `https://api.openai.com/v1` |
| `NEWSLETTER_API_KEY` | API key (or `OPENAI_API_KEY`); optional for local endpoints |
| `NEWSLETTER_FAST_MODEL` | Model for bulk work; required by `newsletter issue` |
| `NEWSLETTER_REASONING_MODEL` | Model for editing; the writer's voice needs it |
| `NEWSLETTER_EMBEDDING_MODEL` | Model for comparing stories (later work) |
| `NEWSLETTER_<TIER>_PRICE_INPUT` / `NEWSLETTER_<TIER>_PRICE_OUTPUT` | USD per million tokens; without them the cost the server reports is used, else the cost is unknown |
| `NEWSLETTER_TIMEOUT` | Seconds per request, default 60 |
| `NEWSLETTER_MAX_RETRIES` | Retries after a timeout, connection error, 408, 409, 429 or 5xx, default 2 |

A tier's model is required only when that tier is used. The API key is never logged or reported.

### The writer's voice

When `YOURVOICE_API_KEY` is set, each summary is rewritten in the writer's own voice. [yourvoice.dev](https://yourvoice.dev) builds a voice profile from samples of the writer's writing, and the reasoning model rewrites the summary to match it. Then a check compares every link, quoted string, number and claim id before and after the rewrite. If one is dropped, changed or added, or the profile or the rewrite fails or times out, the neutral summary is kept and the run report says which one and why. The command then exits 1. Without the key, the issue stays in a neutral voice, and the run report says so.

| Variable | Purpose |
| --- | --- |
| `YOURVOICE_API_KEY` | yourvoice.dev API key; turns the voice on. Sent as `X-API-Key`, never logged or reported |
| `NEWSLETTER_VOICE_SAMPLES` | Folder of the writer's own `.md` and `.txt` files to build the profile from |
| `NEWSLETTER_VOICE_CACHE_DIR` | Where the profile is cached, default `$XDG_CACHE_HOME/newsletter/voice` (`~/.cache/newsletter/voice`). A run builds a new profile only when the samples change; each build costs yourvoice.dev credits |
| `YOURVOICE_BASE_URL` | Default `https://api.yourvoice.dev` |
| `YOURVOICE_TIMEOUT` | Seconds per request, default 60 |

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m newsletter.evals   # eval table from recorded responses, no network
```

The eval replays recorded model answers for past feeds that include planted false claims. `uv run python -m newsletter.evals --live -o evals/results.md` runs the same feeds against the models in your environment and spends credit.

## Contribute

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Share

[![Twitter](https://img.shields.io/badge/X-Tweet-blue)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20project!&url=https://github.com/samestrin/llm-newsletter-generator) [![Facebook](https://img.shields.io/badge/Facebook-Share-blue)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/samestrin/llm-newsletter-generator) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Share-blue)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/samestrin/llm-newsletter-generator)
