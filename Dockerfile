# syntax=docker/dockerfile:1.7

ARG NODE_VERSION=24
ARG RUST_VERSION=1.96

FROM node:${NODE_VERSION}-bookworm-slim AS frontend-builder
WORKDIR /app/web/frontend

COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci

COPY web/frontend/ ./
ENV VITE_API_BASE_URL=
RUN npm run build

FROM rust:${RUST_VERSION}-bookworm AS rust-builder
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY data/ ./data/
COPY crates/ ./crates/
RUN cargo build --locked --release -p llm-infer-cal-web

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app

WORKDIR /app
COPY --from=rust-builder /app/target/release/llm-infer-cal-web /usr/local/bin/llm-infer-cal-web
COPY --from=frontend-builder /app/web/frontend/dist/ /app/static/

ENV LLM_INFER_CAL_WEB_ADDR=0.0.0.0:8080
ENV LLM_INFER_CAL_STATIC_DIR=/app/static

EXPOSE 8080
USER 10001:10001

ENTRYPOINT ["llm-infer-cal-web"]
