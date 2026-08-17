"""Watermarks — the last meeting date each source got through cleanly.

`paddock ingest since` (T14) asks one question: what has changed since we last
succeeded? The watermark is the answer, and the only interesting rule is that it
**never moves backwards**. Backfill and the live pipeline write to the same row: a
2024-25 meeting ingested after this week's would rewind the mark and make the next
`since` run re-ingest a season. So an older date updates `last_run_at` and leaves
`last_race_date` alone.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest

from paddock.db.models import Watermark
from paddock.db.session import session_scope
from paddock.ingest.watermark import advance_watermark, get_watermark

pytestmark = pytest.mark.integration

# Source names no real pipeline uses, so a failed run cannot disturb ingestion.
SOURCE = "test_only_source"
OTHER_SOURCE = "test_only_other"


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    yield
    with session_scope() as session:
        session.query(Watermark).filter(Watermark.source.in_([SOURCE, OTHER_SOURCE])).delete(
            synchronize_session=False
        )


def test_an_unknown_source_has_no_watermark() -> None:
    """Before the first run there is no floor — `since` must mean "everything"."""
    with session_scope() as session:
        assert get_watermark(session, SOURCE) is None


def test_first_success_sets_the_mark() -> None:
    with session_scope() as session:
        advance_watermark(session, SOURCE, dt.date(2026, 4, 26))

    with session_scope() as session:
        assert get_watermark(session, SOURCE) == dt.date(2026, 4, 26)


def test_a_later_meeting_moves_the_mark_forward() -> None:
    with session_scope() as session:
        advance_watermark(session, SOURCE, dt.date(2026, 4, 26))
        advance_watermark(session, SOURCE, dt.date(2026, 5, 3))

    with session_scope() as session:
        assert get_watermark(session, SOURCE) == dt.date(2026, 5, 3)


def test_an_older_meeting_does_not_rewind_the_mark() -> None:
    """Backfilling 2024-25 after this week's meeting must not re-open a season."""
    with session_scope() as session:
        advance_watermark(session, SOURCE, dt.date(2026, 5, 3))
        advance_watermark(session, SOURCE, dt.date(2024, 11, 13))

    with session_scope() as session:
        assert get_watermark(session, SOURCE) == dt.date(2026, 5, 3)


def test_an_older_meeting_still_records_that_the_pipeline_ran() -> None:
    """`last_run_at` answers "is ingestion alive?", which is a different question."""
    with session_scope() as session:
        advance_watermark(session, SOURCE, dt.date(2026, 5, 3))
        first_run = session.get(Watermark, SOURCE).last_run_at  # type: ignore[union-attr]

    with session_scope() as session:
        advance_watermark(session, SOURCE, dt.date(2024, 11, 13))
        assert session.get(Watermark, SOURCE).last_run_at > first_run  # type: ignore[union-attr,operator]


def test_sources_advance_independently() -> None:
    """A stalled sectionals feed must not make the report feed look stalled too."""
    with session_scope() as session:
        advance_watermark(session, SOURCE, dt.date(2026, 5, 3))
        advance_watermark(session, OTHER_SOURCE, dt.date(2026, 4, 26))

    with session_scope() as session:
        assert get_watermark(session, SOURCE) == dt.date(2026, 5, 3)
        assert get_watermark(session, OTHER_SOURCE) == dt.date(2026, 4, 26)
