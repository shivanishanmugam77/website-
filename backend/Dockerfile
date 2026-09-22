# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Unprivileged user: nothing in this project needs root at runtime.
RUN groupadd --system --gid 1001 app \
 && useradd --system --uid 1001 --gid app --home-dir /app --no-create-home app

WORKDIR /app
RUN chown app:app /app \
 && mkdir -p /data/storage /data/models && chown -R app:app /data

# OpenCV (used by the text-recognition engine) needs these shared libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# PyTorch, CPU-only: the default wheels bring ~2 GB of NVIDIA libraries the laptop cannot use.
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt -r requirements-ml.txt

# ---- development image (used by docker-compose): adds test/lint tooling, source is bind-mounted
FROM base AS dev
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY --chown=app:app . .
USER app
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/app", "--no-access-log"]

# ---- production image (default target): runtime dependencies + application only
FROM base AS prod
COPY --chown=app:app app ./app
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini .
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
