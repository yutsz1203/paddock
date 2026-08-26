#!/usr/bin/env bash
#
# Build data/seed/paddock_demo.dump — the demo dataset the repository ships.
#
# The corpus is never touched. The slice is cut from a throwaway copy, so a run
# that fails half-way costs a copy and not an hour of scraping:
#
#   paddock  ──copy──►  paddock_seed  ──prune──►  pg_dump  ──►  data/seed/
#
# The copy is a dump piped into a restore rather than `CREATE DATABASE ... TEMPLATE`.
# A template clone is faster and needs the source to have no other connections, which
# means one forgotten `psql` stops the build. This way is a minute of local I/O and
# never asks anyone to close anything.
#
# Everything that talks to Postgres goes through the compose container, so this
# needs no psql or pg_dump on the host.
#
# Usage: make seed  [MEETINGS=20]

set -euo pipefail

cd "$(dirname "$0")/.."

MEETINGS="${MEETINGS:-20}"
DUMP="data/seed/paddock_demo.dump"

# GitHub warns above 50 MB and refuses above 100 MB. Checked here rather than
# written down, so the slice cannot quietly grow past what a clone can carry.
MAX_MB="${MAX_MB:-50}"

compose() { docker compose "$@"; }
psql_as() { compose exec -T postgres psql -U paddock -v ON_ERROR_STOP=1 "$@"; }

# Read the configured database through the app's own settings, so this follows
# whatever DATABASE_URL says rather than assuming the compose defaults.
read -r SOURCE_DB SEED_URL <<<"$(uv run python -c '
from sqlalchemy.engine import make_url
from paddock.config import Settings

url = make_url(str(Settings().database_url))
seed = url.set(database=f"{url.database}_seed")
print(url.database, seed.render_as_string(hide_password=False))
')"
SEED_DB="${SOURCE_DB}_seed"

echo "==> copying ${SOURCE_DB} to ${SEED_DB}"
psql_as -d postgres -c "DROP DATABASE IF EXISTS \"${SEED_DB}\"" >/dev/null
psql_as -d postgres -c "CREATE DATABASE \"${SEED_DB}\"" >/dev/null
# The page archive is left behind here rather than deleted later: it is 117 MB the
# copy has no use for, and the prune drops it in any case. Both ends of the pipe are
# inside the container, so nothing crosses the docker socket.
compose exec -T postgres bash -c "pg_dump -U paddock --exclude-table-data=fetched_pages \
  '${SOURCE_DB}' | psql -U paddock -q -v ON_ERROR_STOP=1 -d '${SEED_DB}'" >/dev/null

echo "==> cutting the slice to ${MEETINGS} meetings"
DATABASE_URL="${SEED_URL}" uv run paddock demo prune --meetings "${MEETINGS}" --yes

echo "==> dumping to ${DUMP}"
mkdir -p data/seed
# Custom format so the restore can create the schema, the data and the indexes in
# one command, and -Z9 because the file is committed and read far more than written.
compose exec -T postgres pg_dump -U paddock -Fc -Z9 "${SEED_DB}" > "${DUMP}"

psql_as -d postgres -c "DROP DATABASE IF EXISTS \"${SEED_DB}\"" >/dev/null

SIZE_MB=$(( $(wc -c < "${DUMP}") / 1024 / 1024 ))
echo "==> ${DUMP} is ${SIZE_MB} MB"
if [ "${SIZE_MB}" -gt "${MAX_MB}" ]; then
  echo "that is over the ${MAX_MB} MB ceiling. Lower MEETINGS and build it again." >&2
  exit 1
fi
