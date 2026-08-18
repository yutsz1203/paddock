"""Point the whole suite at a database and an address space of its own.

## Why this file exists

Until it did, `uv run pytest` wrote to the same Postgres the real corpus lives in,
and keyed its fixture pages by the same URLs production uses. Both halves of that
went wrong at once during T11's 2025-26 backfill: `test_cli.py` archived
`results_20260423_no_meeting.html` under the production results URLs for 2026-04-26,
`fetch_page` read the archive before the network and trusted it, and a real meeting
was ingested with ten of its eleven races recorded as having no results page. The
teardowns then deleted that meeting and one other from the corpus entirely.

## What it does about it

**A database of its own.** `DATABASE_URL` is rewritten to the configured database
with `_test` appended, created and migrated on first use. So the corpus is not
reachable from a test at all, and `_delete_everything()` has nothing of value to
delete.

**An address space of its own.** `HKJC_BASE_URL` is pointed at an unroutable host, so
`HkjcClient.url_for` — which several tests use deliberately, to key the archive
exactly as production does — cannot produce a real URL. A test database full of
fixture pages is still a hazard the day someone restores it somewhere; pages that
could never be mistaken for HKJC's are not.

Both are needed. The database alone leaves fixture bodies wearing real URLs; the base
URL alone still lets a teardown delete real meetings.

## Ordering

The environment is set at import, not in a fixture, because `get_settings` and
`get_engine` are both `lru_cache`d and the first test module to import `paddock`
would otherwise freeze the real URL in place. Any cache already built is cleared
below. Provisioning is deferred to the first test that actually asks for a database,
so `make test-unit` still needs no Postgres.
"""

from __future__ import annotations

import os
import pathlib

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = pathlib.Path(__file__).parent.parent

# Unroutable by RFC 6761: a request that escapes the test doubles fails to resolve
# rather than reaching anyone.
TEST_BASE_URL = "https://hkjc.test.invalid"


def _test_database_url() -> str:
    """The configured database with `_test` appended, or it unchanged if already so.

    Derived rather than hardcoded so it follows whatever host, port and credentials
    are configured — CI sets `DATABASE_URL` to its own service container, and this
    has to land beside it rather than somewhere else entirely.
    """
    # Read directly rather than through the cached `get_settings`, so this works no
    # matter what has been imported already.
    from paddock.config import Settings

    url = make_url(str(Settings().database_url))
    if not (url.database or "").endswith("_test"):
        url = url.set(database=f"{url.database}_test")
    # `str(URL)` renders the password as `***`. Harmless when printing, silently
    # fatal when the result is used to connect.
    return url.render_as_string(hide_password=False)


_URL = _test_database_url()
os.environ["DATABASE_URL"] = _URL
os.environ["HKJC_BASE_URL"] = TEST_BASE_URL

# Anything imported before this point cached the real settings. Drop those caches so
# the redirect applies to the whole run rather than to whatever imported last.
from paddock.config import get_settings  # noqa: E402
from paddock.db.session import get_engine, get_session_factory  # noqa: E402

get_settings.cache_clear()
get_engine.cache_clear()
get_session_factory.cache_clear()

_provisioned = False


def _provision() -> None:
    """Create the test database if absent and migrate it to head. Runs once."""
    global _provisioned
    if _provisioned:
        return

    url = make_url(_URL)
    assert url.database and url.database.endswith("_test"), url.database

    # `CREATE DATABASE` cannot run inside a transaction, and cannot run from a
    # connection to the database being created — hence the maintenance connection.
    admin_url = url.set(database="postgres").render_as_string(hide_password=False)
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        exists = connection.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database}
        )
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    admin.dispose()

    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    # Both set explicitly so the suite does not depend on being run from the repo
    # root, and so alembic reads the redirected URL rather than alembic.ini's.
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", _URL)
    command.upgrade(config, "head")

    _provisioned = True


@pytest.fixture(autouse=True)
def _database(request: pytest.FixtureRequest) -> None:
    """Provision before the first test that needs a database, and not before.

    Keyed on the `integration` marker rather than on a fixture request, because the
    integration tests reach for `session_scope()` directly and ask for nothing.
    """
    if request.node.get_closest_marker("integration") is not None:
        _provision()
