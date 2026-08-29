.PHONY: help install up down db lint typecheck test test-unit test-model test-ui ui seed demo image deploy-config check

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
	uv run mypy src/ app/

test-unit:  ## Unit tests only (no database)
	uv run pytest -m "not integration"

# The suite runs against `<database>_test`, which tests/conftest.py creates and
# migrates on first use — never against the corpus. `make db` is for the real one.
test:  ## All tests (requires Postgres; uses its own _test database)
	uv run pytest

# Deselected by default so CI never depends on Hugging Face being up. Run this
# after touching anything in paddock/embed — it is the only proof that a Chinese
# query finds its English source, which is the whole reason for bge-m3.
test-model:  ## Embedding tests against the real bge-m3 (~2.2 GB download)
	uv sync --extra embed
	uv run pytest -m model

# Also deselected by default, but for a different reason than `model`: streamlit is
# an optional extra and the rest of the suite runs on a bare `uv sync`. CI installs
# every extra and runs this in a step of its own.
test-ui:  ## Streamlit demo tests (needs the `ui` extra)
	uv sync --extra ui
	uv run pytest -m ui

ui:  ## Run the Streamlit demo (needs `paddock serve` on :8000)
	uv run streamlit run app/streamlit_app.py

# Cuts data/seed/paddock_demo.dump from a throwaway clone of the corpus. The corpus
# itself is never written to. Only needed after a backfill widens what should ship.
seed:  ## Rebuild the committed demo dataset from the local corpus
	./scripts/seed.sh

# Restores data/seed/ into a database of its own and starts the API and the demo.
# No HKJC request is made. The corpus database is not touched.
demo:  ## Run the demo from the committed dataset
	./scripts/demo.sh

# Builds what the Oracle instance runs. `--platform` is explicit because the box is
# aarch64 and an image built for the wrong architecture fails at `docker run`, on
# the box, after the whole transfer.
image:  ## Build the deployment image
	docker build --platform linux/arm64 -t paddock:latest .

# Merges the base stack with the deploy overlay and prints the result. Run it after
# editing either file: `tests/unit/test_deploy_config.py` asserts the properties
# that matter, and this shows what Compose actually made of them.
deploy-config:  ## Show the merged deployment configuration
	PADDOCK_DOMAIN=$${PADDOCK_DOMAIN:-example.invalid} ACME_EMAIL=$${ACME_EMAIL:-you@example.invalid} \
	  docker compose -f docker-compose.yml -f docker-compose.deploy.yml config

check: lint typecheck test  ## Everything
