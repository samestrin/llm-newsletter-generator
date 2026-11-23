# Public demo on the `demos` droplet

The public demo of the newsletter generator runs on the `demos` droplet at `https://newsletter.samestrin.com` (the host name is proposed, matching the other demos' `<app>.samestrin.com`; the owner confirms it on fleet-deploy's PR), behind the droplet's Traefik. A visitor brings a **demo code** or **their own model key**, fills the form, watches every step live, and reads a draft whose claims are checked against their sources. Nothing sends mail. This page covers what this repo ships for it: [`docker-compose.demo.yml`](../../docker-compose.demo.yml).

The droplet's half is **fleet-deploy's**, not this repo's: the deploy-target entry, the Traefik router for the demo host, the demo-code entries, and the gateway model mapping. This repo ships the compose file, this page, and the shape test. The demo itself also needs the demo-mode API (demo code / visitor key access, `GET /api/providers`, `POST /api/feeds/check`, the demo limits and per-browser draft scoping), which is epic 08.1's code; the compose file is the deploy shape and its CI check validates the file, not a running demo.

## What the compose file runs

- Two services, both built from this repo's `Dockerfile` on the droplet. No source is mounted into them.
  - `api` (target `api`) starts with `newsletter serve --seed-sample`: at start it writes the committed `open-models-week` sample as one draft, once, which powers **"See an example"**. A restart adds no second copy and never touches an existing draft. The seeded draft needs no model key.
  - `web` (target `web`) is the built demo page behind nginx, which forwards `/api/` to `api:8000` unchanged (`web/nginx.conf`).
- **No host port.** Traefik reaches `web` on the external Docker network `proxy`, where it also answers as `newsletter`, so the gateway's router points at `http://newsletter:80`. `api` stays on the compose's internal network; only nginx reaches it. (The droplet's edge network and router are fleet-deploy's half.)
- **Demo mode is on** (`NEWSLETTER_DEMO=1`). A run needs a demo code or a visitor key; the server does not spend a key of its own on a visitor's run.
- **No model key in the file.** The gateway key is read from `.env.demo` on the droplet (below). The demo gateway URL and the demo model names are set in the file; they are not secrets.
- Health: `api` keeps the same healthcheck as `docker-compose.yml` (`GET /api/health`); `web` fetches `/` through nginx and requires the built page's `<div id="root">`, then checks `/api/health` through the same nginx, so one check proves the page and the proxied API together. fleet-deploy's receiver waits for `web`'s healthcheck and rolls back to the previous images if it fails.
- `restart: unless-stopped`. Drafts live in the `drafts` volume (`/data`), which survives deploys.
- The project is named `llm-newsletter-generator` and no service sets `image:`, so the images are `llm-newsletter-generator-api` and `llm-newsletter-generator-web`. The receiver tags each `:latest` as `:rollback` before a deploy.

## The env it reads

The non-secret demo settings are set in the compose file's `environment:` block:

| Variable | Value | What it does |
| --- | --- | --- |
| `NEWSLETTER_DEMO` | `1` | Demo mode: a run needs a demo code or a visitor key; the demo limits and per-browser draft scoping apply. |
| `NEWSLETTER_DEMO_BASE_URL` | `http://litellm:4000/v1` | The droplet's model gateway (litellm). A demo code is sent here as its bearer. |
| `NEWSLETTER_DEMO_FAST_MODEL` | `demo-fast` | The gateway's demo model for the bulk work (summaries, claims, checks). |
| `NEWSLETTER_DEMO_REASONING_MODEL` | `demo-reasoning` | The gateway's demo model for the editor pass. |
| `NEWSLETTER_DEMO_EMBEDDING_MODEL` | `demo-embedding` | The gateway's demo model for grouping stories by meaning. |
| `NEWSLETTER_API_ORIGINS` | `https://newsletter.samestrin.com` | The demo's https origin. The page calls the API from that origin; a write from any other origin is refused. |
| `NEWSLETTER_REASONING_TIMEOUT` | `300` | Seconds per reasoning (editor) call; a reasoning model can take minutes. |
| `NEWSLETTER_RUN_TIMEOUT` | `1800` | Seconds a whole run may take. It must fit the run, not one call: the editor alone can retry up to 3 × 300 s = 900 s, leaving headroom for feed fetch, article read, embedding, claims and checks over the demo's ≤18 items. |

The demo model names are the ones the gateway must serve for this app; fleet-deploy's gateway entry maps them (the droplet's litellm config is its half). The gateway URL is the litellm service on the droplet's internal models network.

### The gateway key: `.env.demo`

The gateway key comes only from `.env.demo` beside the compose file on the droplet (mode 600). It is never in git (`.gitignore` lists it), and it is never written in the compose file. Compose will not start without it.

| Variable | What to set |
| --- | --- |
| `NEWSLETTER_API_KEY` | The gateway key the droplet holds for this app. It is a secret: it is redacted from everything the API stores, returns or logs, and it is never a visitor's demo code or key. |

A visitor's run does not use this key: a demo code goes to the gateway as its own bearer, and a visitor key goes only to that visitor's provider. The key in `.env.demo` is the server's own gateway credential, kept on the droplet and out of git.

## The demo-code handout

Demo codes are litellm virtual keys on the droplet gateway, each with its own budget, expiry and rate limits. fleet-deploy makes, changes and revokes them (`droplet/litellm/codes.py`, entries in `droplet/demo-codes.yaml`; the owner approves each code). A new code's value is written once to a 0600 file for the owner to hand out, never printed to a log. To try the demo, a visitor pastes the code into the key panel ("Demo code"); the page sends it only as the `X-Demo-Code` header, and the API forwards it to the gateway as the bearer for that run's model calls. The code is never stored or logged by this app. Removing an entry revokes the code; a used-up or expired code is refused by the gateway with a plain message.

## Visitor keys

A visitor may instead bring their own key for OpenAI, OpenRouter, Alibaba or Synthetic (the key panel's provider list). The key is sent only as the `X-Model-Key` header with its provider (`X-Model-Preset`), is used for **that one run's** model calls to that provider, and is **never stored or logged**: it is not written to the run record, the draft, the report, any response or any log line, and it is redacted from anything the API stores or returns. It is held in memory for the run only. A provider with no embedding model (Synthetic) still completes; the run says stories are not grouped by meaning.

## The limits

In demo mode the API enforces (epic 08.1):

- **3 items per feed** — a run reads at most the first 3 items of each feed.
- **6 feeds** — a run takes at most 6 feed URLs; a 7th is refused.
- **One run at a time** — a second `POST /api/runs` while one is running answers 409 with a `retry_after`; the page shows "Someone else is making one right now. Try again in a minute."
- `POST /api/feeds/check` is capped at 30 checks a minute (429 with `retry_after` over the cap).

Each demo code also carries its own budget, expiry and requests/tokens-per-minute at the gateway, and the gateway has a monthly backstop budget.

## The abuse stance

The per-IP rate limit is the droplet's Traefik **`ratelimit-models`** middleware in **fleet-deploy**, not code in this repo. It covers `POST /api/runs` and `POST /api/feeds/check` **only once fleet-deploy adds a newsletter router that uses it**. Until that router exists, those two routes have no per-IP Traefik limit; the protections in force are the app-level ones above (one run at a time, the feed-check cap, the demo limits) and each demo code's own budget and rate limits at the gateway. This page says so until the router is in place.

## Check it

Validate the file anywhere (CI's `compose` job does this with a dummy settings file):

```bash
printf 'NEWSLETTER_API_KEY=ci-dummy\n' > .env.demo
docker compose -f docker-compose.demo.yml config -q
rm .env.demo
```

`tests/test_demo_compose.py` guards the shape the deploy depends on: no `ports:`, no secret in the file, demo mode on, the gateway key only from `.env.demo`, the demo gateway and model names set, an https origin, a reasoning timeout of 300 and a run timeout that fits it, a data volume, `serve --seed-sample`, and `web`'s healthcheck.