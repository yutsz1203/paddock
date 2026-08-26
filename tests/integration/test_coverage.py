"""What the corpus holds, and the `/coverage` endpoint that reports it.

The demo must state its data range plainly (T22). A hardcoded banner is the obvious
way to do that and the wrong one: the live pipeline (T14) adds a meeting every few
days from 6 September, and a string frozen at "through 15 July 2026" then tells a
visitor something false about a system whose whole claim is that it does not. So the
range is a query, and the UI renders whatever it answers.

**These tests own the meetings table.** Coverage is an aggregate over every meeting,
so a leftover row from another module changes the answer. The suite runs against its
own database (see `tests/conftest.py`), which is what makes a full wipe here safe.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from paddock.api.main import app, get_llm_dependency
from paddock.db.coverage import corpus_coverage
from paddock.db.models import Meeting
from paddock.db.session import session_scope

pytestmark = pytest.mark.integration

OPENING = dt.date(2024, 9, 8)  # 2024-25
MID = dt.date(2025, 3, 12)  # 2024-25
CLOSING = dt.date(2026, 7, 15)  # 2025-26


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_every_meeting()
    yield
    _delete_every_meeting()


def _delete_every_meeting() -> None:
    with session_scope() as session:
        session.query(Meeting).delete(synchronize_session=False)


def _seed(*days: dt.date) -> None:
    with session_scope() as session:
        for day in days:
            session.add(Meeting(race_date=day, racecourse="ST"))


def test_coverage_reports_the_first_and_last_meeting() -> None:
    _seed(MID, CLOSING, OPENING)

    with session_scope() as session:
        coverage = corpus_coverage(session)

    assert coverage.meetings == 3
    assert coverage.first_date == OPENING
    assert coverage.last_date == CLOSING


def test_coverage_names_the_seasons_that_have_meetings() -> None:
    """Named, not spanned. A season between the first and last meeting with nothing
    in it must not be claimed — that is the sentence a visitor would check."""
    _seed(OPENING, CLOSING)

    with session_scope() as session:
        coverage = corpus_coverage(session)

    assert coverage.seasons == ["2024-25", "2025-26"]


def test_an_empty_corpus_reports_no_range_rather_than_a_wrong_one() -> None:
    with session_scope() as session:
        coverage = corpus_coverage(session)

    assert coverage.meetings == 0
    assert coverage.first_date is None
    assert coverage.last_date is None
    assert coverage.seasons == []


def test_the_endpoint_serves_what_the_banner_needs() -> None:
    _seed(OPENING, MID, CLOSING)

    with TestClient(app) as client:
        response = client.get("/coverage")

    assert response.status_code == 200
    assert response.json() == {
        "meetings": 3,
        "first_date": OPENING.isoformat(),
        "last_date": CLOSING.isoformat(),
        "seasons": ["2024-25", "2025-26"],
    }


def test_the_endpoint_needs_no_llm() -> None:
    """A demo whose key has expired must still be able to say what data it holds.

    The override is inert by design — `/coverage` declares no provider dependency —
    and that is the property being pinned: nothing on this path can be broken by a
    missing key.
    """
    _seed(OPENING)
    app.dependency_overrides[get_llm_dependency] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.get("/coverage")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["meetings"] == 1
