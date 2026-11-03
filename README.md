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

`newsletter issue` reads RSS or Atom feeds, summarizes each item with the fast model and writes the issue as Markdown with a link per item, then a run report listing any feed, entry or model call that failed and what each step cost. It exits 1 when anything failed, so a partial issue is never reported as a success. Use `--cache-dir` to keep fetched feeds between runs, `--date` to set the issue date and `--per-feed N` to summarize only the first N items of each feed (each item is one model call).

```bash
uv run newsletter archive --demo --site site --per-feed 5 https://example.com/feed.xml
```

`newsletter archive` drafts the same issue (in the writer's voice too, when `YOURVOICE_API_KEY` is set; the daily workflow sets none) and builds it into a static site in `--site`: an index and one page per issue. Only demo mode exists so far, so `--demo` is required. A demo issue is approved by a record marked `automated-demo` instead of a person's sha256 of the text; that record is valid for archive pages only and export refuses it. Every demo page carries a visible "Automated demo" label, and every summary is marked as unchecked.

### Daily archive

`.github/workflows/daily.yml` runs `newsletter archive --demo` every day (and by hand from the Actions tab) on five public AI news feeds, five items each, and uploads the site as the `archive-site` workflow artifact. It has no deploy step and never runs on pull requests. To turn it on, set the secret `NEWSLETTER_API_KEY` and the repository variable `NEWSLETTER_FAST_MODEL` (and `NEWSLETTER_BASE_URL` for a non-OpenAI endpoint).

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

### The API

`newsletter serve` runs an HTTP API for drafting and reviewing issues. A run reads the feeds, then the editor checks each claim against its source and writes the issue, then the voice pass rewrites the editor's prose, then the issue is rendered. Each run is a background job with a status and a timeout. Its result is a draft: the rendered Markdown, the flagged claims and the run report, kept as a JSON file in the data dir, so it survives a restart. Approving a draft stores the sha256 of its text. Any edit clears the approval.

```bash
export NEWSLETTER_API_TOKEN=$(openssl rand -hex 32)
uv run newsletter serve            # http://127.0.0.1:8000
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Authorization: Bearer $NEWSLETTER_API_TOKEN" -H "Origin: http://127.0.0.1:8000" \
  -H "Content-Type: application/json" -d '{"feeds": ["https://example.com/feed.xml"]}'
```

| Endpoint | Does |
| --- | --- |
| `GET /api/health` | `{"ok": true}`; the only call without the token |
| `POST /api/runs` | Starts a run from `{"feeds": [...], "title", "topic", "date"}`; 202 with the run, 409 while another run is going |
| `GET /api/runs/{id}` | The run: `queued`, `running`, `done` with its `draft_id`, or `error` with why. A run past its timeout, or cut off by a restart, ends as an `error` and leaves no draft |
| `GET /api/drafts` | Every draft, newest first, without its text |
| `GET /api/drafts/{id}` | The draft: `text`, `flags`, `report`, `approval` and `approved` |
| `PUT /api/drafts/{id}` | Replaces the text from `{"text": "..."}` and clears the approval |
| `POST /api/drafts/{id}/approve` | Approves the text as it is now. Send `{}`, or `{"sha256": "..."}` of the text you read to get a 409 if it changed since |

Every call but the health check needs `Authorization: Bearer $NEWSLETTER_API_TOKEN`. A write also needs an `Origin` from `NEWSLETTER_API_ORIGINS` and a JSON object sent as `application/json`; anything else is refused with 401, 403, 415 or 400. Without a token, the server will not start. The token and the model and voice keys never appear in a response, an error body, a stored file or a log.

| Variable | Purpose |
| --- | --- |
| `NEWSLETTER_API_TOKEN` | The token every call must send; required |
| `NEWSLETTER_API_HOST` | Bind address, default `127.0.0.1` (this machine only). `--host` overrides it |
| `NEWSLETTER_API_PORT` | Port, default 8000. `--port` overrides it |
| `NEWSLETTER_API_ORIGINS` | Comma-separated origins allowed to write, default `http://127.0.0.1:<port>,http://localhost:<port>` |
| `NEWSLETTER_DATA_DIR` | Where drafts and runs are kept, default `$XDG_DATA_HOME/newsletter` (`~/.local/share/newsletter`) |
| `NEWSLETTER_RUN_TIMEOUT` | Seconds a run may take, default 900 |

### The review page

`web/` is a page (React and Vite) for reviewing drafts. Paste the API token, pick a draft, and read its text with its flagged claims and its run report: failed feeds, dropped stories and claims, editor and voice problems, and each model call with its tokens and cost. Edit the text and save it, which clears any approval, then approve. Approve is off while an edit is unsaved. It sends the sha256 of the text you read, so if the draft changed since, the API refuses it and the page says so; Reload fetches the new text. The draft text is always shown as text, never as HTML, and only http and https sources are links. The token stays in the page's memory, so a reload asks for it again.

```bash
export NEWSLETTER_API_TOKEN=$(openssl rand -hex 32)
NEWSLETTER_API_ORIGINS=http://localhost:5173 uv run newsletter serve
cd web && npm ci && npm run dev    # http://localhost:5173, /api goes to 127.0.0.1:8000
```

The page's origin must be in `NEWSLETTER_API_ORIGINS`, or the API refuses its writes. `NEWSLETTER_API_URL` points the dev server's `/api` somewhere else. `VITE_API_URL` at build time makes the built page call the API there directly instead of on its own origin.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m newsletter.evals   # eval table from recorded responses, no network
cd web && npm ci && npm run build && npm test   # the review page: build, unit tests with coverage
npx playwright install chromium && npm run test:e2e   # browser test against a fake API
```

The eval replays recorded model answers for past feeds that include planted false claims. `uv run python -m newsletter.evals --live -o evals/results.md` runs the same feeds against the models in your environment and spends credit.

## Contribute

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Share

[![Twitter](https://img.shields.io/badge/X-Tweet-blue)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20project!&url=https://github.com/samestrin/llm-newsletter-generator) [![Facebook](https://img.shields.io/badge/Facebook-Share-blue)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/samestrin/llm-newsletter-generator) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Share-blue)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/samestrin/llm-newsletter-generator)
