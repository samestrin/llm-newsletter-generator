# Private review deploy on embassy

The owner's review copy of the newsletter generator runs on embassy at `https://newsletter.estr.in` (the host name is proposed; the owner confirms it on fleet-deploy's PR), behind Traefik and an Authelia login. Every route, the review page and the API alike, answers only after that login. The deploy itself (the deploy-target entry, Traefik route, Authelia rule, first start) is fleet-deploy's: its `orchestrator/deploy-targets.yaml` gains a `llm-newsletter-generator` entry (kind `embassy-compose`, `compose_file: docker-compose.embassy.yml`, `service: web`, `health_path: /api/health`), and `docs/review-stack.md` and `docs/apply/install-gateway.md` have the usage and gateway steps. This page covers what this repo ships for it: [`docker-compose.embassy.yml`](../../docker-compose.embassy.yml).

## What the compose file runs

- Two services, both built from this repo's `Dockerfile` on embassy. No source is mounted into them.
  - `api` (target `api`) starts with `newsletter serve --seed-sample`: at start it writes the committed `open-models-week` sample as one draft, once. A restart adds no second copy and never touches an existing draft. The seeded draft needs no model key.
  - `web` (target `web`) is the built review page behind nginx, which forwards `/api/` to `api:8000` unchanged (`web/nginx.conf`).
- No host port. Traefik reaches `web` on the external Docker network `proxy`, where it also answers as `newsletter`, so the gateway's router points at `http://newsletter:80`. `api` stays on the compose's internal network; only nginx reaches it.
- No model server and no model key in the file. To run a *new* issue on the review instance (beyond the seeded sample), add the model settings below to `.env.embassy`; the seeded draft works without them.
- Health: `api` keeps the same healthcheck as `docker-compose.yml` (`GET /api/health`); `web` fetches `/` through nginx and requires the built page's `<div id="root">` (a missing or empty build fails), then checks `/api/health` through the same nginx, so one check proves the page and the proxied API together. fleet-deploy's receiver waits for `web`'s healthcheck and rolls back to the previous images if it fails.
- `restart: unless-stopped`. Drafts live in the `drafts` volume (`/data`), which survives deploys.
- The project is named `llm-newsletter-generator` and no service sets `image:`, so the images are `llm-newsletter-generator-api` and `llm-newsletter-generator-web`. The receiver tags each `:latest` as `:rollback` before a deploy.

## Settings: `.env.embassy`

Every setting comes from `.env.embassy` beside the compose file on embassy (`~/docker/llm-newsletter-generator/.env.embassy`, mode 600). It is never in git (`.gitignore` lists it), and the receiver refuses a deploy archive that carries one. Compose will not start without it.

| Variable | What to set |
| --- | --- |
| `NEWSLETTER_API_TOKEN` | The bearer token every API call but the health check must send. Generate one (`openssl rand -hex 32`); it is never written anywhere else. |
| `NEWSLETTER_API_ORIGINS` | The review host's https origin: `https://newsletter.estr.in`. The page calls the API from that origin, and a write from any other origin is refused. |

Optional, only to run new issues on the review instance: any model setting from the [Configuration](../../README.md#configuration) table, for example `NEWSLETTER_BASE_URL`, `NEWSLETTER_API_KEY`, `NEWSLETTER_FAST_MODEL`, `NEWSLETTER_REASONING_MODEL` and `NEWSLETTER_EMBEDDING_MODEL` pointing at the fleet litellm with this app's own key. Without them the instance still serves and edits the seeded draft; starting a new run fails with the model settings' own error.

## Check it

Validate the file anywhere (CI's `compose` job does this with a dummy settings file):

```bash
touch .env.embassy
docker compose -f docker-compose.embassy.yml config -q
```

`tests/test_embassy_compose.py` guards the shape the receiver depends on: no `ports:`, no secret in the file, settings only from `.env.embassy`, `serve --seed-sample`, and `web`'s healthcheck.

Through Traefik, without a login, every path on the review host answers with a redirect to Authelia or 401, never with the app. With the owner's login the review page loads with the seeded draft, and `https://newsletter.estr.in/api/health` answers `{"ok": true}`.
