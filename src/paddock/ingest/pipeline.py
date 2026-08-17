"""Ingestion orchestration: bookkeeping, idempotent upserts, atomicity.

The parsers (T4, T5) are pure functions over a page body and the archive (`pages.py`)
supplies that body without hitting the network twice. What is left is the part that
touches the database, and it has to hold three guarantees at once:

**Ingesting a meeting twice changes nothing the second time.** Every write is an
upsert keyed on the natural uniqueness the schema already declares — `(race_date,
racecourse)`, `(meeting_id, race_no)`, `(race_id, horse_id)`. Not "delete then
rewrite", which would churn primary keys and orphan the chunks that cite them.

**A meeting that fails half-way leaves nothing behind.** One transaction per meeting.
Nine races written and the tenth malformed means zero races written.

**But the bookkeeping survives that rollback.** The run record and the archived pages
are written in their own transactions, so a failure is visible afterwards and the
pages it cost are not paid for again. Bookkeeping that rolls back with the work it
describes would only ever record successes.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager

from paddock.db.models import IngestRun
from paddock.db.session import session_scope
from paddock.ingest.date_guard import FallbackDetectedError


class RunRecord:
    """Handle on the open `ingest_runs` row, for outcomes other than ok/failed."""

    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        self.status: str | None = None
        self.error: str | None = None

    def pending(self, reason: str) -> None:
        """Close as `report_pending` — the page is not published *yet*.

        Distinct from a failure because a retry is the right response and an alert is
        not: HKJC posts the stewards' report hours after the results (T14).
        """
        self.status = "report_pending"
        self.error = reason


@contextmanager
def record_run(source: str, race_date: dt.date | None) -> Iterator[RunRecord]:
    """Record one ingestion attempt, whatever happens to it.

    Writes `running` immediately, so a process killed mid-meeting is visible as a run
    that never finished, then `ok`, `failed`, `fallback_detected` or `report_pending`
    on the way out. Each write is its own transaction — see the module docstring.

    Exceptions are recorded and re-raised, never swallowed: the caller decides
    whether one bad meeting stops a backfill.
    """
    started_at = dt.datetime.now(dt.UTC)
    with session_scope() as session:
        row = IngestRun(source=source, race_date=race_date, status="running", started_at=started_at)
        session.add(row)
        session.flush()
        record = RunRecord(row.id)

    try:
        yield record
    except BaseException as error:
        # A backfill over generated candidate dates expects two rejections for every
        # hit, so recording those as failures would bury the ones that need a human.
        # MeetingHeaderMissingError is deliberately *not* in this category: it means
        # the guard can no longer tell real pages from substituted ones.
        status = "fallback_detected" if isinstance(error, FallbackDetectedError) else "failed"
        _close(record.run_id, status=status, error=str(error))
        raise

    _close(record.run_id, status=record.status or "ok", error=record.error)


def _close(run_id: int, *, status: str, error: str | None) -> None:
    with session_scope() as session:
        row = session.get(IngestRun, run_id)
        if row is None:  # pragma: no cover — only reachable if someone truncates the table
            return
        row.status = status
        row.error = error
        row.finished_at = dt.datetime.now(dt.UTC)
