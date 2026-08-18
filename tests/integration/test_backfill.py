"""Backfilling a season: many meetings, most of which are not meetings at all.

Ingesting one meeting (T10) assumed an operator who knew the date and the venue.
A season has neither of those to hand:

**Nobody types 88 venues.** The current-season index is a list of dates. The prior
season has no index at all, only generated Wed/Sat/Sun candidates. So the venue is
read off the report page, and these tests drive `ingest_meeting` without a course.

**Two candidate dates in three are not meetings.** The guard raises on each one, and
a backfill that stopped there would need 140 restarts to cross 2024-25. So a rejected
date is recorded and the walk continues, as does a meeting that simply will not parse
— its own transaction already rolled back, and `ingest_runs` has the date. The single
exception is a report with no meeting header: past that point the guard cannot tell a
real page from a substituted one, so continuing would write real data under wrong
dates. That one stops the run.

No network: `RecordingFetcher` serves the same fixture meeting under whatever date is
asked for, which is exactly what HKJC's fallback does and what the guard exists to
catch.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Iterator

import pytest
from sqlalchemy import select
from tests.doubles import RecordingFetcher

from paddock.db.models import FetchedPage, IngestRun, Meeting, Watermark
from paddock.db.session import session_scope
from paddock.ingest import pipeline
from paddock.ingest.backfill import backfill
from paddock.ingest.date_guard import MeetingHeaderMissingError
from paddock.ingest.incident_report import ReportParseError
from paddock.ingest.pipeline import ingest_meeting
from paddock.ingest.watermark import INCIDENT_REPORT

pytestmark = pytest.mark.integration

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "html"

RACE_DATE = dt.date(2026, 4, 26)  # Sunday, Sha Tin, 11 races
HV_DATE = dt.date(2025, 3, 12)  # Wednesday, Happy Valley, 9 races
FALLBACK_DATE = dt.date(2026, 4, 23)  # Thursday — HK never races on one
RACES_IN_CARD = 11


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_everything()
    yield
    _delete_everything()


def _delete_everything() -> None:
    days = (RACE_DATE, HV_DATE, FALLBACK_DATE)
    with session_scope() as session:
        for day in days:
            session.query(Meeting).filter(Meeting.race_date == day).delete(
                synchronize_session=False
            )
            session.query(IngestRun).filter(IngestRun.race_date == day).delete(
                synchronize_session=False
            )
            session.query(FetchedPage).filter(FetchedPage.url.in_(_urls_for(day))).delete(
                synchronize_session=False
            )
        session.query(Watermark).filter(Watermark.source == INCIDENT_REPORT).delete(
            synchronize_session=False
        )


def _urls_for(day: dt.date) -> list[str]:
    """Every URL a meeting on `day` could be archived under, at either course."""
    url_for = RecordingFetcher().url_for
    urls = [url_for(pipeline.REPORT_PATH, pipeline.report_params(day))]
    for course in ("ST", "HV"):
        for race_no in range(1, RACES_IN_CARD + 1):
            urls.append(
                url_for(pipeline.RESULTS_PATH, pipeline.results_params(day, course, race_no))
            )
    for race_no in range(1, RACES_IN_CARD + 1):
        urls.append(url_for(pipeline.SECTIONALS_PATH, pipeline.sectionals_params(day, race_no)))
    return urls


def _serve_meeting(
    client: RecordingFetcher, day: dt.date, report_fixture: str, course: str, races: int
) -> RecordingFetcher:
    """One meeting's pages. Results and sectionals exist for Race 1 only."""
    client.serve(
        pipeline.REPORT_PATH,
        pipeline.report_params(day),
        (FIXTURES / report_fixture).read_text(),
    )
    empty = (FIXTURES / "results_20260423_no_meeting.html").read_text()
    for race_no in range(1, races + 1):
        client.serve(
            pipeline.RESULTS_PATH,
            pipeline.results_params(day, course, race_no),
            (FIXTURES / "results_20260426_ST_R1.html").read_text() if race_no == 1 else empty,
        )
        client.serve(
            pipeline.SECTIONALS_PATH,
            pipeline.sectionals_params(day, race_no),
            (FIXTURES / "sectional_20260426_R1.html").read_text() if race_no == 1 else empty,
        )
    return client


# ── Reading the venue off the page ──────────────────────────────────────────────


def test_a_meeting_ingested_without_a_course_reads_it_from_the_report() -> None:
    """The 2025-26 index is 88 dates and no venues; this is where the venue comes from."""
    client = _serve_meeting(
        RecordingFetcher(), RACE_DATE, "report_20260426_valid.html", "ST", RACES_IN_CARD
    )

    result = ingest_meeting(client, RACE_DATE)

    assert result.races == RACES_IN_CARD
    with session_scope() as session:
        meeting = session.scalar(select(Meeting).where(Meeting.race_date == RACE_DATE))
        assert meeting is not None
        assert meeting.racecourse == "ST"


def test_the_other_racecourse_is_read_just_as_well() -> None:
    """Both venues from the same code path — a default would silently be right half
    the time, which is the hardest kind of wrong to notice."""
    client = _serve_meeting(
        RecordingFetcher(), HV_DATE, "report_20250312_prior_season.html", "HV", 9
    )

    ingest_meeting(client, HV_DATE)

    with session_scope() as session:
        meeting = session.scalar(select(Meeting).where(Meeting.race_date == HV_DATE))
        assert meeting is not None
        assert meeting.racecourse == "HV"


def test_the_detected_venue_decides_which_results_pages_are_fetched() -> None:
    """Not just the stored column: the venue is in every results URL, so getting it
    wrong would ingest a full card with no result attached to any of it."""
    client = _serve_meeting(
        RecordingFetcher(), HV_DATE, "report_20250312_prior_season.html", "HV", 9
    )

    ingest_meeting(client, HV_DATE)

    assert any("Racecourse=HV" in url for url in client.requests)
    assert not any("Racecourse=ST" in url for url in client.requests)


def test_an_explicit_course_still_wins() -> None:
    """`paddock ingest meeting --course` predates this and stays authoritative."""
    client = _serve_meeting(
        RecordingFetcher(), RACE_DATE, "report_20260426_valid.html", "ST", RACES_IN_CARD
    )

    ingest_meeting(client, RACE_DATE, "ST")

    with session_scope() as session:
        meeting = session.scalar(select(Meeting).where(Meeting.race_date == RACE_DATE))
        assert meeting is not None
        assert meeting.racecourse == "ST"


# ── Walking a season ────────────────────────────────────────────────────────────


def _season_client() -> RecordingFetcher:
    """Two real meetings with a non-meeting Thursday between them."""
    client = RecordingFetcher()
    _serve_meeting(client, RACE_DATE, "report_20260426_valid.html", "ST", RACES_IN_CARD)
    _serve_meeting(client, HV_DATE, "report_20250312_prior_season.html", "HV", 9)
    client.serve(
        pipeline.REPORT_PATH,
        pipeline.report_params(FALLBACK_DATE),
        (FIXTURES / "report_20260423_fallback.html").read_text(),
    )
    return client


SEASON = [HV_DATE, FALLBACK_DATE, RACE_DATE]


def test_every_meeting_in_the_range_is_ingested() -> None:
    report = backfill(_season_client(), SEASON)

    assert [o.race_date for o in report.ingested] == [HV_DATE, RACE_DATE]
    with session_scope() as session:
        stored = set(
            session.scalars(select(Meeting.race_date).where(Meeting.race_date.in_(SEASON)))
        )
        assert stored == {HV_DATE, RACE_DATE}


def test_a_date_the_guard_rejects_does_not_stop_the_walk() -> None:
    """Two candidate dates in three are not meetings. A backfill that stopped on the
    first one would need a hundred restarts to cross a season."""
    report = backfill(_season_client(), SEASON)

    assert [o.race_date for o in report.rejected] == [FALLBACK_DATE]
    assert len(report.ingested) == 2, "the date after the rejection was still ingested"


def test_a_rejected_date_is_left_for_a_human_to_look_at() -> None:
    """`ingest_runs` is the log: a rejection count that climbs is how a markup change
    announces itself, and each row names the date to go and check by hand."""
    backfill(_season_client(), SEASON)

    with session_scope() as session:
        run = session.scalar(select(IngestRun).where(IngestRun.race_date == FALLBACK_DATE))
        assert run is not None
        assert run.status == "fallback_detected"
        assert "2026-07-15" in (run.error or ""), "the meeting HKJC served instead"


def test_nothing_is_written_for_a_rejected_date() -> None:
    backfill(_season_client(), SEASON)

    with session_scope() as session:
        assert session.scalar(select(Meeting).where(Meeting.race_date == FALLBACK_DATE)) is None


# ── Restarting ──────────────────────────────────────────────────────────────────


def test_a_meeting_already_stored_is_not_ingested_twice() -> None:
    """A run that dies at meeting 60 of 88 is resumed, not repeated. The upserts make
    a repeat harmless, but 60 meetings of pointless writes is an hour either way."""
    backfill(_season_client(), SEASON)

    second = backfill(_season_client(), SEASON)

    assert [o.race_date for o in second.skipped] == [HV_DATE, RACE_DATE]
    assert second.ingested == []


def test_a_skipped_meeting_records_no_second_run() -> None:
    backfill(_season_client(), SEASON)
    backfill(_season_client(), SEASON)

    with session_scope() as session:
        runs = list(session.scalars(select(IngestRun).where(IngestRun.race_date == RACE_DATE)))
        assert len(runs) == 1


def test_refresh_re_ingests_what_is_already_there() -> None:
    """For the meetings whose stewards' report HKJC corrected after publication."""
    backfill(_season_client(), SEASON)

    second = backfill(_season_client(), SEASON, refresh=True)

    assert [o.race_date for o in second.ingested] == [HV_DATE, RACE_DATE]
    assert second.skipped == []


# ── When something is actually wrong ────────────────────────────────────────────


def test_one_bad_meeting_does_not_cost_the_other_eighty_seven() -> None:
    """A single meeting that will not parse is recorded and stepped over. It cannot
    corrupt anything — the meeting transaction rolled back — and the report says so."""
    client = _season_client()
    client.fail(
        pipeline.REPORT_PATH,
        pipeline.report_params(HV_DATE),
        ReportParseError("no table.rirr elements"),
    )

    report = backfill(client, SEASON)

    assert [o.race_date for o in report.failed] == [HV_DATE]
    assert [o.race_date for o in report.ingested] == [RACE_DATE]


def test_a_guard_that_can_no_longer_tell_stops_everything() -> None:
    """The one failure that must not be stepped over. Without the header the guard
    cannot separate a real page from a substituted one, so every date after this is
    unverifiable — and ingesting them would write real data under wrong dates."""
    client = _season_client()
    client.pages[client.url_for(pipeline.REPORT_PATH, pipeline.report_params(HV_DATE))] = (
        "<html><body>no header here</body></html>"
    )

    with pytest.raises(MeetingHeaderMissingError):
        backfill(client, SEASON)

    with session_scope() as session:
        assert session.scalar(select(Meeting).where(Meeting.race_date == RACE_DATE)) is None


# ── Watching it run ─────────────────────────────────────────────────────────────


def test_each_outcome_is_announced_as_it_happens() -> None:
    """The real run is 1-2 hours long. A report printed at the end of it is a report
    nobody can tell from a hang."""
    seen: list[tuple[dt.date, str]] = []

    backfill(_season_client(), SEASON, on_outcome=lambda o: seen.append((o.race_date, o.status)))

    assert seen == [
        (HV_DATE, "ingested"),
        (FALLBACK_DATE, "rejected"),
        (RACE_DATE, "ingested"),
    ]


def test_the_report_counts_what_was_written() -> None:
    report = backfill(_season_client(), SEASON)

    assert report.meetings == 2
    assert report.races == RACES_IN_CARD + 9
    assert report.comments > 0
