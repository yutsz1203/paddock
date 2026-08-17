"""Every ingestion attempt is recorded, especially the ones that fail.

The point of this table is the failures, and that puts one constraint on the design:
**the run record must not share the meeting's transaction.** A meeting that fails
half-way rolls back — that is T10's "no partial meeting" — and if the record rode
along it would roll back too, leaving a failure that never happened. So the record
is written in its own session, before and after the work.

A row is written as `running` up front rather than only at the end, so a process
killed mid-meeting is visible afterwards as a run that never finished.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import select

from paddock.db.models import IngestRun
from paddock.db.session import session_scope
from paddock.ingest.date_guard import FallbackDetectedError
from paddock.ingest.pipeline import record_run

pytestmark = pytest.mark.integration

SOURCE = "test_only_source"
RACE_DATE = dt.date(2026, 4, 26)


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    yield
    with session_scope() as session:
        session.query(IngestRun).filter(IngestRun.source == SOURCE).delete(
            synchronize_session=False
        )


def _runs() -> list[IngestRun]:
    with session_scope() as session:
        return list(
            session.scalars(
                select(IngestRun).where(IngestRun.source == SOURCE).order_by(IngestRun.id)
            )
        )


def test_a_successful_run_is_recorded_ok() -> None:
    with record_run(SOURCE, RACE_DATE):
        pass

    (run,) = _runs()
    assert run.status == "ok"
    assert run.race_date == RACE_DATE
    assert run.finished_at is not None
    assert run.error is None


def test_a_run_is_visible_while_it_is_still_running() -> None:
    """Otherwise a process killed mid-meeting leaves no trace at all."""
    with record_run(SOURCE, RACE_DATE):
        (in_flight,) = _runs()
        assert in_flight.status == "running"
        assert in_flight.finished_at is None


def test_a_failure_is_recorded_with_its_error_and_re_raised() -> None:
    with pytest.raises(RuntimeError, match="markup moved"), record_run(SOURCE, RACE_DATE):
        raise RuntimeError("markup moved")

    (run,) = _runs()
    assert run.status == "failed"
    assert "markup moved" in (run.error or "")
    assert run.finished_at is not None


def test_a_failure_survives_the_rollback_of_the_work_it_describes() -> None:
    """The record is written in its own transaction — that is the whole design."""
    with pytest.raises(RuntimeError), record_run(SOURCE, RACE_DATE), session_scope() as session:
        session.add(IngestRun(source=SOURCE, status="ok", started_at=dt.datetime.now(dt.UTC)))
        raise RuntimeError("meeting blew up")

    runs = _runs()
    assert [r.status for r in runs] == ["failed"], "the caller's own write must have rolled back"


def test_a_served_fallback_is_recorded_as_such_not_as_a_failure() -> None:
    """A date with no meeting is expected during backfill; a broken parser is not."""
    with (
        pytest.raises(FallbackDetectedError),
        record_run(SOURCE, dt.date(2026, 4, 23)),
    ):
        raise FallbackDetectedError(requested=dt.date(2026, 4, 23), served=RACE_DATE)

    (run,) = _runs()
    assert run.status == "fallback_detected"
    assert "2026-04-26" in (run.error or ""), "the date actually served is the useful detail"


def test_the_caller_can_close_a_run_as_pending() -> None:
    """The stewards' report lands hours after the results — not yet is not failed."""
    with record_run(SOURCE, RACE_DATE) as run:
        run.pending("incident report not published yet")

    (row,) = _runs()
    assert row.status == "report_pending"
    assert row.error == "incident report not published yet"
