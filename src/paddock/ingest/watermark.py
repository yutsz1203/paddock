"""How far each source has been ingested successfully.

`paddock ingest since` asks one question — what has changed since we last succeeded?
— and this is where the answer lives. One row per source, so a stalled sectionals
feed does not make the incident report look stalled too.

**The mark only moves forward.** Backfill (T11) and the live pipeline (T14) write to
the same row, and backfill walks *backwards* through 2024-25. If an older meeting
lowered the mark, the next `since` run would re-ingest a season. So an older date is
recorded as "the pipeline ran" and nothing more.

**It is set on success only.** A failed meeting leaves the mark where it was, which
is what makes a retry pick that meeting up again rather than skip past it. The
failure itself is recorded in `ingest_runs`; this table is not an error log.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from paddock.db.models import Watermark

# The sources a watermark is kept for. Named here rather than passed as free strings
# so a typo is an import error instead of a silently separate, never-advancing row.
INCIDENT_REPORT = "incident_report"


def get_watermark(session: Session, source: str) -> dt.date | None:
    """The newest meeting date `source` has ingested cleanly, or None if never."""
    row = session.get(Watermark, source)
    return row.last_race_date if row else None


def advance_watermark(session: Session, source: str, race_date: dt.date) -> None:
    """Record a successful ingest of `race_date`, moving the mark forward only.

    `last_run_at` is always updated — "did ingestion run?" and "how far has it got?"
    are different questions, and backfill answers only the first.
    """
    now = dt.datetime.now(dt.UTC)

    # One statement rather than read-then-write: two schedulers running at once would
    # otherwise race between the check and the update. GREATEST does the "forward
    # only" rule in the database, and ignores NULLs, so a row with no date yet takes
    # the incoming one rather than staying NULL.
    statement = insert(Watermark).values(source=source, last_race_date=race_date, last_run_at=now)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[Watermark.source],
            set_={
                "last_race_date": func.greatest(Watermark.last_race_date, race_date),
                "last_run_at": now,
            },
        )
    )
    session.flush()
