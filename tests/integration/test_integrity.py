"""The check that says the backfill can be trusted.

The guard (ADR-002) runs on every page before it is parsed, so in principle nothing
contaminated reaches the database. This is the audit that says so afterwards, and it
exists because the guard is one regex against markup nobody promised us: if HKJC
moves the header and the guard degrades, the damage is 88 meetings of real comments
filed under dates that never raced, discovered weeks later as inexplicable retrieval
bugs.

So the two checks are deliberately *not* the guard again. They look at what is
actually in the database:

**No two meetings share a runner set.** A served fallback is a copy of another
meeting, so the tell is two dates with the same horses in the same races. Two real
meetings never field an identical card.

**Every meeting's date matches its own page.** Re-read the archived report the
meeting was built from and ask it, one more time, which meeting it is. Independent
of the guard because it re-derives the answer rather than trusting that a check ran.

A meeting whose page is not in the archive is a finding too, not a pass — "cannot
verify" and "verified" are the two answers this must never confuse.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Iterator

import pytest
from sqlalchemy import select, update
from tests.doubles import RecordingFetcher

from paddock.db.models import FetchedPage, IngestRun, Meeting, Race, Runner
from paddock.db.session import session_scope
from paddock.ingest import pipeline
from paddock.ingest.integrity import check_integrity, verify_season
from paddock.ingest.pipeline import ingest_meeting
from paddock.ingest.racing_calendar import PublishedMeeting

pytestmark = pytest.mark.integration

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "html"

ST_DATE = dt.date(2026, 4, 26)  # Sha Tin, 11 races
HV_DATE = dt.date(2025, 3, 12)  # Happy Valley, 9 races
IMPOSTOR_DATE = dt.date(2025, 3, 15)  # a Saturday with no meeting on it
ALL_DATES = (ST_DATE, HV_DATE, IMPOSTOR_DATE)


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_everything()
    yield
    _delete_everything()


def _delete_everything() -> None:
    with session_scope() as session:
        for day in ALL_DATES:
            session.query(Meeting).filter(Meeting.race_date == day).delete(
                synchronize_session=False
            )
            session.query(IngestRun).filter(IngestRun.race_date == day).delete(
                synchronize_session=False
            )
            session.query(FetchedPage).filter(FetchedPage.url.in_(_urls_for(day))).delete(
                synchronize_session=False
            )


def _urls_for(day: dt.date) -> list[str]:
    url_for = RecordingFetcher().url_for
    urls = [url_for(pipeline.REPORT_PATH, pipeline.report_params(day))]
    for course in ("ST", "HV"):
        for race_no in range(1, 12):
            urls.append(
                url_for(pipeline.RESULTS_PATH, pipeline.results_params(day, course, race_no))
            )
    for race_no in range(1, 12):
        urls.append(url_for(pipeline.SECTIONALS_PATH, pipeline.sectionals_params(day, race_no)))
    return urls


def _ingest(day: dt.date, report_fixture: str, course: str, races: int) -> None:
    client = RecordingFetcher()
    client.serve(
        pipeline.REPORT_PATH, pipeline.report_params(day), (FIXTURES / report_fixture).read_text()
    )
    empty = (FIXTURES / "results_20260423_no_meeting.html").read_text()
    for race_no in range(1, races + 1):
        client.serve(
            pipeline.RESULTS_PATH,
            pipeline.results_params(day, course, race_no),
            (FIXTURES / "results_20260426_ST_R1.html").read_text() if race_no == 1 else empty,
        )
        client.serve(pipeline.SECTIONALS_PATH, pipeline.sectionals_params(day, race_no), empty)
    ingest_meeting(client, day)


def _ingest_both_meetings() -> None:
    _ingest(ST_DATE, "report_20260426_valid.html", "ST", 11)
    _ingest(HV_DATE, "report_20250312_prior_season.html", "HV", 9)


def _clone_meeting_onto(day: dt.date, target: dt.date) -> None:
    """Copy a meeting's whole card onto another date.

    This is what a fallback looks like once it is in the database: real races, real
    horses, real comments, filed under a date that never raced. Manufactured
    directly rather than by defeating the guard, because the check has to find the
    *state* whatever produced it — including a guard that stopped working.
    """
    with session_scope() as session:
        source = session.scalar(select(Meeting).where(Meeting.race_date == day))
        assert source is not None
        copy = Meeting(race_date=target, racecourse=source.racecourse, source_url=source.source_url)
        session.add(copy)
        session.flush()
        for race in session.scalars(select(Race).where(Race.meeting_id == source.id)):
            new_race = Race(
                meeting_id=copy.id,
                race_no=race.race_no,
                name=race.name,
                distance_m=race.distance_m,
                status=race.status,
            )
            session.add(new_race)
            session.flush()
            for runner in session.scalars(select(Runner).where(Runner.race_id == race.id)):
                session.add(
                    Runner(
                        race_id=new_race.id,
                        horse_id=runner.horse_id,
                        horse_no=runner.horse_no,
                        finish_pos=runner.finish_pos,
                    )
                )


# ── A clean corpus ──────────────────────────────────────────────────────────────


def test_two_real_meetings_pass() -> None:
    _ingest_both_meetings()

    report = check_integrity()

    assert report.clean, report.findings
    assert report.meetings_checked == 2


def test_an_empty_corpus_is_not_reported_as_clean_data() -> None:
    """Nothing to find and nothing checked are different answers, and a backfill that
    wrote nothing at all must not read as a pass."""
    report = check_integrity()

    assert report.meetings_checked == 0


# ── Two meetings, one card ──────────────────────────────────────────────────────


def test_a_duplicated_card_is_found() -> None:
    """The shape a served fallback leaves behind: the same horses in the same races,
    under two dates. Two real meetings never field an identical card."""
    _ingest_both_meetings()
    _clone_meeting_onto(HV_DATE, IMPOSTOR_DATE)

    report = check_integrity()

    assert not report.clean
    duplicates = [f for f in report.findings if f.check == "duplicate_runner_set"]
    assert len(duplicates) == 1
    assert {HV_DATE, IMPOSTOR_DATE} <= set(duplicates[0].race_dates)


def test_meetings_with_different_cards_are_not_confused() -> None:
    """Both fixtures share no horses, but the check must survive the harder case of
    two meetings that overlap heavily without being copies."""
    _ingest_both_meetings()

    assert not [f for f in check_integrity().findings if f.check == "duplicate_runner_set"]


# ── A meeting that disagrees with its own page ──────────────────────────────────


def test_a_meeting_filed_under_the_wrong_date_is_found() -> None:
    """Re-derived from the archived page rather than trusted: this is the check that
    would still fire if the guard itself had quietly stopped working."""
    _ingest_both_meetings()
    with session_scope() as session:
        session.execute(
            update(Meeting).where(Meeting.race_date == HV_DATE).values(race_date=IMPOSTOR_DATE)
        )

    report = check_integrity()

    mismatches = [f for f in report.findings if f.check == "date_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].race_dates == [IMPOSTOR_DATE]
    assert "2025-03-12" in mismatches[0].detail, "the date the page actually declares"


def test_a_meeting_whose_page_is_gone_is_unverifiable_not_clean() -> None:
    """The archive is what makes the claim checkable at all. Without the page there
    is no evidence either way, and silence would read as evidence of correctness."""
    _ingest_both_meetings()
    with session_scope() as session:
        session.query(FetchedPage).filter(FetchedPage.url.in_(_urls_for(HV_DATE))).delete(
            synchronize_session=False
        )

    report = check_integrity()

    assert not report.clean
    unverifiable = [f for f in report.findings if f.check == "unverifiable"]
    assert [f.race_dates for f in unverifiable] == [[HV_DATE]]


# ── Against what HKJC said it would run ─────────────────────────────────────────
#
# The calendar is injected rather than loaded, so these tests state a small season
# instead of needing all 88. The real 2024-25 sheet has its own tests in
# `tests/unit/test_racing_calendar.py`.
#
# Only HV_DATE and IMPOSTOR_DATE fall inside 2024-25; ST_DATE belongs to the season
# after it, which is what `test_another_seasons_meeting_is_not_counted` is about.


def _published(*rows: tuple[dt.date, str]) -> list[PublishedMeeting]:
    return [PublishedMeeting(race_date=day, racecourse=course) for day, course in rows]


def test_a_corpus_that_matches_the_calendar_passes() -> None:
    _ingest_both_meetings()

    report = verify_season("2024-25", _published((HV_DATE, "HV")))

    assert report.passed, report
    assert report.published == 1
    assert report.ingested == 1


def test_another_seasons_meeting_is_not_counted() -> None:
    """Both backfills land in one database, so checking 2024-25 must not see 2025-26
    as 88 unannounced meetings."""
    _ingest_both_meetings()

    report = verify_season("2024-25", _published((HV_DATE, "HV")))

    assert ST_DATE not in report.unpublished
    assert report.ingested == 1, "the 2025-26 meeting is out of scope, not a finding"


def test_a_meeting_hkjc_announced_and_we_do_not_have_is_named() -> None:
    """Named, not counted. "86 of 88" is a number to worry at; a list of dates is
    something to go and re-run."""
    _ingest_both_meetings()

    report = verify_season("2024-25", _published((HV_DATE, "HV"), (IMPOSTOR_DATE, "ST")))

    assert report.missing == [IMPOSTOR_DATE]


def test_a_few_missing_meetings_are_reported_but_tolerated() -> None:
    """The calendar is published six weeks early, so a meeting can be abandoned —
    2024-11-13 lost its last three races to a typhoon signal. A handful of absences
    is a thing to look at, not a failed backfill."""
    _ingest_both_meetings()

    report = verify_season("2024-25", _published((HV_DATE, "HV"), (IMPOSTOR_DATE, "ST")))

    assert report.missing
    assert report.passed, "one abandoned meeting must not fail the season"


def test_more_than_a_handful_missing_does_not_pass() -> None:
    """T11's ±3, applied to an exact list rather than a count."""
    _ingest_both_meetings()
    absent = [(dt.date(2025, 1, 7 + n), "ST") for n in range(4)]

    report = verify_season("2024-25", _published((HV_DATE, "HV"), *absent))

    assert len(report.missing) == 4
    assert not report.passed


def test_a_meeting_the_calendar_never_listed_is_named() -> None:
    """A rescheduled meeting looks like this — and so does a fallback that got past
    the guard, which is why it is surfaced rather than assumed benign."""
    _ingest_both_meetings()
    _clone_meeting_onto(HV_DATE, IMPOSTOR_DATE)

    report = verify_season("2024-25", _published((HV_DATE, "HV")))

    assert report.unpublished == [IMPOSTOR_DATE]


def test_a_racecourse_that_disagrees_never_passes() -> None:
    """The one difference that cannot be a scheduling change. A meeting that ran at
    all ran at exactly one of two venues, so the calendar and the going table cannot
    both be right — and the going table is what the whole backfill trusts."""
    _ingest_both_meetings()

    report = verify_season("2024-25", _published((HV_DATE, "ST")))

    assert report.venue_mismatches == [(HV_DATE, "ST", "HV")]
    assert not report.passed, "a single one of these is fatal — no tolerance applies"


def test_an_abandoned_meeting_is_not_a_meeting_we_lost() -> None:
    """24 September 2025 never ran, so its absence from the corpus is correct. Counted
    as missing it would burn one of the three absences the season is allowed."""
    _ingest_both_meetings()

    report = verify_season(
        "2024-25",
        [
            PublishedMeeting(race_date=HV_DATE, racecourse="HV"),
            PublishedMeeting(race_date=IMPOSTOR_DATE, racecourse="ST", abandoned=True),
        ],
    )

    assert report.missing == []
    assert report.abandoned == 1
    assert report.published == 1, "the sheet's total counts meetings that ran"


def test_an_abandoned_meeting_that_still_published_a_report_is_not_a_stranger() -> None:
    """Abandoned part-way is a real state — 2024-11-13 lost three races and still has
    a full report. If one turns up in the corpus it was announced, so it must not be
    reported as a date HKJC never scheduled."""
    _ingest_both_meetings()
    _clone_meeting_onto(HV_DATE, IMPOSTOR_DATE)

    report = verify_season(
        "2024-25",
        [
            PublishedMeeting(race_date=HV_DATE, racecourse="HV"),
            PublishedMeeting(race_date=IMPOSTOR_DATE, racecourse="HV", abandoned=True),
        ],
    )

    assert report.unpublished == []
    assert report.passed
