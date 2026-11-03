# Two images from one file: `api` (newsletter serve) and `web` (the built review page behind
# nginx, which also forwards /api to the api service). See docker-compose.yml.

FROM python:3.12-slim AS api
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy NEWSLETTER_API_HOST=0.0.0.0 NEWSLETTER_DATA_DIR=/data
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev
RUN useradd --system --uid 10001 newsletter && mkdir /data && chown newsletter /data
USER newsletter
VOLUME /data
EXPOSE 8000
CMD ["/app/.venv/bin/newsletter", "serve"]

FROM node:22-slim AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM nginx:1.27-alpine AS web
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-build /web/dist /usr/share/nginx/html
EXPOSE 80
