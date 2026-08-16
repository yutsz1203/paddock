.PHONY: help install up down db lint typecheck test test-unit check

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

check: lint typecheck test  ## Everything
