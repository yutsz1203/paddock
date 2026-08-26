#!/usr/bin/env bash
#
# Run the demo from a clean clone, with no HKJC network access.
#
#   data/seed/paddock_demo.dump  ──restore──►  paddock_demo  ──►  API :8000 ──►  UI :8501
#
# The restore goes into a database of its own — `<configured>_demo` — and never into
# the corpus. That matters on a developer's machine, where the configured database
# holds two seasons and about an hour of scraping that no dump in this repository
# can put back.
#
# What this still reaches the network for, and why neither is HKJC: the Python
# dependencies, and bge-m3 from Hugging Face (~2.2 GB) the first time a question is
# asked. The racing data comes entirely from the committed dump.
#
# Usage: make demo  [API_PORT=8000] [UI_PORT=8501]

set -euo pipefail

cd "$(dirname "$0")/.."

DUMP="data/seed/paddock_demo.dump"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
API_HOST=127.0.0.1

if [ ! -f "${DUMP}" ]; then
  echo "${DUMP} is missing. It ships with the repository; run \`make seed\` to rebuild" >&2
  echo "it from a local corpus." >&2
  exit 1
fi

# embed for the query encoder, agent for the LLM client and the graph, ui for
# Streamlit. All three are optional extras, so a bare `uv sync` cannot run the demo.
echo "==> installing dependencies (this pulls torch, ~2 GB, on a first run)"
uv sync --extra embed --extra agent --extra ui

echo "==> starting Postgres"
docker compose up -d postgres
until docker compose exec -T postgres pg_isready -U paddock -d paddock >/dev/null 2>&1; do
  sleep 1
done

read -r DEMO_DB DEMO_URL <<<"$(uv run python -c '
from sqlalchemy.engine import make_url
from paddock.config import Settings

url = make_url(str(Settings().database_url))
demo = url.set(database=f"{url.database}_demo")
print(demo.database, demo.render_as_string(hide_password=False))
')"

psql_as() { docker compose exec -T postgres psql -U paddock -v ON_ERROR_STOP=1 "$@"; }

echo "==> restoring ${DUMP} into ${DEMO_DB}"
# Terminating connections is safe here and only here: this database is rebuilt from
# a committed file on every run, so nothing in it is anyone's work.
psql_as -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
  WHERE datname = '${DEMO_DB}' AND pid <> pg_backend_pid()" >/dev/null
psql_as -d postgres -c "DROP DATABASE IF EXISTS \"${DEMO_DB}\"" >/dev/null 2>&1
psql_as -d postgres -c "CREATE DATABASE \"${DEMO_DB}\"" >/dev/null
docker compose exec -T postgres pg_restore -U paddock -d "${DEMO_DB}" --no-owner < "${DUMP}"

echo "==> what the demo holds"
DATABASE_URL="${DEMO_URL}" uv run paddock demo report

if ! DATABASE_URL="${DEMO_URL}" uv run python -c '
from paddock.config import get_settings
from paddock.llm.provider import build_llm

build_llm(get_settings())
' 2>/dev/null; then
  # A warning, not a failure. The banner, the source cards and the data-range
  # answer are all worth seeing without a key; only generation needs one.
  echo "!!  no API key for the configured LLM provider. The demo will start and state"
  echo "!!  its data range, and every question will report the missing key. Set one in"
  echo "!!  .env — see .env.example."
fi

echo "==> starting the API on :${API_PORT}"
DATABASE_URL="${DEMO_URL}" uv run paddock serve --host "${API_HOST}" --port "${API_PORT}" &
API_PID=$!
trap 'kill "${API_PID}" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 30); do
  if curl -sf "http://${API_HOST}:${API_PORT}/health" >/dev/null; then break; fi
  sleep 1
done
if ! curl -sf "http://${API_HOST}:${API_PORT}/health" >/dev/null; then
  echo "the API did not answer /health within 30 s" >&2
  exit 1
fi

echo "==> starting the demo on :${UI_PORT} — Ctrl-C stops both"
echo "    the first question loads bge-m3 and takes about a minute; later ones do not"
UI_API_BASE_URL="http://${API_HOST}:${API_PORT}" \
  uv run streamlit run app/streamlit_app.py --server.port "${UI_PORT}"
