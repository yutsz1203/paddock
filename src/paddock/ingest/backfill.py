"""Walking a season, when most of the dates are not meetings.

`ingest_meeting` handles one date and raises on anything it does not like. A season
is 88 real meetings out of roughly 140 candidate dates, run over one to two hours,
and it needs the opposite disposition: keep going, write down what happened, and be
resumable when the laptop lid closes at meeting 60.

## What stops the run and what does not

    rejected   guard says HKJC served another meeting     → recorded, continue
    failed     this meeting will not parse                → recorded, continue
    skipped    already stored                             → continue
    ingested   written                                    → continue

    MeetingHeaderMissingError                             → stop

Only the last one stops it, and the reason is asymmetric damage. A rejection costs
nothing — the date had no meeting, and two in three candidates are like that. A
failed meeting costs one meeting, and its own transaction has already rolled back,
so there is nothing bad in the database to find later. But a report with no meeting
header means the guard can no longer distinguish a real page from a substituted one
(ADR-002), and every date after it would be ingested unverified. That writes real
data under wrong dates — the one outcome this project has a whole module to prevent.

## Resuming

A restart skips the dates already stored, so the run picks up where it stopped. The
dates it *did* check and reject are re-checked, and that is free: `fetch_page`
archived the report page before the guard looked at it, so the second pass reads
from Postgres and makes no request. That is the property T10's page archive was
bought for, and it is what makes an interrupted 2024-25 backfill cheap to finish.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import select

from paddock.db.models import Meeting
from paddock.db.session import session_scope
from paddock.ingest.date_guard import FallbackDetectedError, MeetingHeaderMissingError
from paddock.ingest.incident_report import ReportParseError
from paddock.ingest.pages import PageFetcher
from paddock.ingest.pipeline import MeetingIngest, ingest_meeting
from paddock.ingest.results import ResultsParseError

# What a scraper is allowed to find wrong with one meeting without stopping the run.
# MeetingHeaderMissingError is pointedly absent — see the module docstring.
_SURVIVABLE = (ReportParseError, ResultsParseError)


@dataclass(frozen=True)
class Outcome:
    """What became of one date."""

    race_date: dt.date
    status: str
    """ingested | rejected | skipped | failed."""
    detail: str | None = None
    """The reason, for everything except an ingest. Printed and kept in the report,
    because "12 rejected" is a number to shrug at and twelve dated reasons are not."""
    ingest: MeetingIngest | None = None


@dataclass
class BackfillReport:
    outcomes: list[Outcome] = field(default_factory=list)

    def _with(self, status: str) -> list[Outcome]:
        return [outcome for outcome in self.outcomes if outcome.status == status]

    @property
    def ingested(self) -> list[Outcome]:
        return self._with("ingested")

    @property
    def rejected(self) -> list[Outcome]:
        """Dates the guard turned away. Expected in bulk on a prior season; a single
        one in a run driven by HKJC's own index means something moved."""
        return self._with("rejected")

    @property
    def skipped(self) -> list[Outcome]:
        return self._with("skipped")

    @property
    def failed(self) -> list[Outcome]:
        return self._with("failed")

    @property
    def meetings(self) -> int:
        return len(self.ingested)

    @property
    def races(self) -> int:
        return sum(o.ingest.races for o in self.ingested if o.ingest)

    @property
    def runners(self) -> int:
        return sum(o.ingest.runners for o in self.ingested if o.ingest)

    @property
    def comments(self) -> int:
        return sum(o.ingest.comments for o in self.ingested if o.ingest)

    @property
    def races_without_results(self) -> list[tuple[dt.date, list[int]]]:
        """Per meeting, the races whose results page had nothing. T10 returns these
        rather than discarding the meeting; this is where a backfill surfaces them."""
        return [
            (o.race_date, o.ingest.races_without_results)
            for o in self.ingested
            if o.ingest and o.ingest.races_without_results
        ]


def backfill(
    client: PageFetcher,
    dates: Iterable[dt.date],
    *,
    refresh: bool = False,
    on_outcome: Callable[[Outcome], None] | None = None,
) -> BackfillReport:
    """Ingest every date in `dates`, in the order given.

    Args:
        refresh: re-ingest dates already stored, and re-fetch their pages. For the
            handful of meetings whose stewards' report HKJC corrected afterwards.
        on_outcome: called with each `Outcome` as it happens. The real run is one to
            two hours long, and a report printed at the end of it is indistinguishable
            from a hang.

    Raises:
        MeetingHeaderMissingError: the guard can no longer verify a page, so no date
            after this one can be trusted. Everything written up to here stands.
    """
    report = BackfillReport()
    stored = _already_stored(dates if isinstance(dates, Sequence) else list(dates))

    for race_date in dates:
        outcome = _one_date(client, race_date, stored=stored, refresh=refresh)
        report.outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    return report


def _one_date(
    client: PageFetcher, race_date: dt.date, *, stored: set[dt.date], refresh: bool
) -> Outcome:
    if race_date in stored and not refresh:
        return Outcome(race_date, "skipped", "already ingested")

    try:
        ingest = ingest_meeting(client, race_date, refresh=refresh)
    except FallbackDetectedError as error:
        # Not a failure: this is the guard doing its job on a date that never raced,
        # which is the majority outcome when candidates were generated rather than
        # indexed. `record_run` has already written the row a human would review.
        return Outcome(race_date, "rejected", str(error))
    except _SURVIVABLE as error:
        return Outcome(race_date, "failed", f"{type(error).__name__}: {error}")

    return Outcome(race_date, "ingested", ingest=ingest)


def _already_stored(dates: Sequence[dt.date]) -> set[dt.date]:
    """Which of these dates are meetings we already have.

    One query for the whole run rather than one per date: 88 round trips to learn
    "no" 88 times is the kind of thing that is invisible locally and slow on the box.
    """
    if not dates:
        return set()
    with session_scope() as session:
        return set(session.scalars(select(Meeting.race_date).where(Meeting.race_date.in_(dates))))


__all__ = [
    "BackfillReport",
    "MeetingHeaderMissingError",
    "Outcome",
    "backfill",
]
