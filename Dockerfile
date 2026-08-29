# One image, two processes.
#
# The API and the demo are separate processes by design (spec §5), so the image
# runs whichever it is told to. Building them separately would mean two builds of
# the same 2 GB of dependencies for the sake of a different final line.
#
# Built for linux/arm64: the box is an Oracle VM.Standard.A1.Flex, which is
# aarch64. The torch wheel that `--extra embed` pulls is CPU-only on that platform,
# which is what we want — there is no GPU, and bge-m3 embeds one query at a time.

FROM python:3.13-slim-bookworm AS base

# libgomp1 is torch's OpenMP runtime. The slim image does not carry it, and without
# it the import fails at run time rather than at build time — so the container
# starts, answers /health, and dies on the first question.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    # Written to a mounted volume, so the 2.2 GB model survives a restart. Without
    # this the first question after every deploy waits on a download.
    HF_HOME=/models

WORKDIR /app

# Dependencies before source, so editing a Python file does not re-resolve or
# re-download torch. `--no-install-project` because the project itself is the layer
# below.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra embed --extra agent --extra ui

COPY src/ ./src/
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./
RUN uv sync --frozen --no-dev --extra embed --extra agent --extra ui

# Not root. The model cache and the page archive are the only things it writes, and
# neither needs privilege.
RUN useradd --create-home --uid 10001 paddock \
    && mkdir -p /models \
    && chown -R paddock:paddock /models /app
USER paddock

EXPOSE 8000 8501

CMD ["paddock", "serve", "--host", "0.0.0.0", "--port", "8000"]
