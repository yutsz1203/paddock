.PHONY: help install up down db lint typecheck test test-unit test-model check

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

up:  ## Start Postgres
	docker compose up -d postgres
	@until docker compose exec -T postgres pg_isready -U paddock -d paddock >/dev/null 2>&1; do sleep 1; done
	@echo "postgres ready"

down:  ## Stop services
	docker compose down

db:  ## Apply migrations
	uv run alembic upgrade head

lint:  ## Lint and format
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Type check
	uv run mypy src/

test-unit:  ## Unit tests only (no database)
	uv run pytest -m "not integration"

test:  ## All tests (requires Postgres)
	uv run pytest

# Deselected by default so CI never depends on Hugging Face being up. Run this
# after touching anything in paddock/embed — it is the only proof that a Chinese
# query finds its English source, which is the whole reason for bge-m3.
test-model:  ## Embedding tests against the real bge-m3 (~2.2 GB download)
	uv sync --extra embed
	uv run pytest -m model

check: lint typecheck test  ## Everything
