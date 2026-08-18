"""The suite must not be able to damage a real corpus.

Written after it did. During T11's 2025-26 backfill this suite was run against the
same database, and two meetings came out of it wrong: `test_cli.py` archives fixture
bodies keyed by `HkjcClient().url_for(...)`, which builds a **production** URL, and
`fetch_page` reads the archive before the network and trusts whatever it finds. The
backfill was served `results_20260423_no_meeting.html` for ten races of a real
meeting and recorded ten missing results pages. Teardown then deleted both meetings
outright.

Two independent things have to be true for that to be impossible, and neither is
sufficient alone:

**A test can never reach the corpus.** The suite runs against its own database, so
a stray `DELETE` has nothing of value to delete.

**A fixture body can never wear a real URL.** Even inside a test database, a page
stored under `https://racing.hkjc.com/...` is a landmine: `pg_dump` it, restore it,
point ingestion at it, and the archive serves a fixture to production. The base URL
is overridden during tests so the two address spaces cannot overlap at all.

These tests are the guard on both. They are cheap and they assert a property of the
whole run, so they earn their place by failing loudly the day someone constructs a
real client in a test.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select

from paddock.config import get_settings
from paddock.db.models import FetchedPage
from paddock.db.session import get_engine, session_scope
from paddock.ingest import pipeline
from paddock.ingest.http import HkjcClient

pytestmark = pytest.mark.integration

PRODUCTION_HOST = "racing.hkjc.com"


def test_the_suite_runs_against_its_own_database() -> None:
    """The single assertion that would have prevented the incident.

    Anything else in this file can be worked around by a determined test; this one
    means the corpus is not reachable from here at all.
    """
    database = get_engine().url.database

    assert database is not None
    assert database.endswith("_test"), (
        f"tests are pointed at {database!r} — refusing to run against a real corpus"
    )


def test_a_test_cannot_build_a_production_url() -> None:
    """Several tests construct a real `HkjcClient` to key the page archive exactly as
    production does. That is the point of them — and it means the base URL is the only
    thing keeping the two address spaces apart."""
    with HkjcClient() as client:
        url = client.url_for(
            pipeline.RESULTS_PATH, pipeline.results_params(dt.date(2026, 4, 26), "ST", 1)
        )

    assert PRODUCTION_HOST not in url
    assert PRODUCTION_HOST not in get_settings().hkjc_base_url


def test_no_page_in_the_test_archive_wears_a_production_url() -> None:
    """Scans whatever the run has accumulated so far. Ordering makes this stronger
    the later it runs, and it costs one indexed count either way."""
    with session_scope() as session:
        offenders = session.scalar(
            select(func.count())
            .select_from(FetchedPage)
            .where(FetchedPage.url.contains(PRODUCTION_HOST))
        )

    assert offenders == 0, (
        f"{offenders} archived pages are keyed by a real HKJC URL — a restore of this "
        "database would serve fixtures to production ingestion"
    )
