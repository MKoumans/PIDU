FROM python:3.13.7 AS base
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app
WORKDIR /app

############################
# Development dependencies #
############################
FROM base AS deps-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc curl libpq-dev && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/
# The environment lives outside /app so that bind-mounting the checkout does not shadow it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

###########
# Dev run #
###########
FROM deps-dev AS dev
COPY . /app/
