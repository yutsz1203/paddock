"""The post-backfill audit: is what we ingested actually what HKJC published?

The date guard (ADR-002) already refuses a substituted page before it is parsed, so
this should never find anything. That is the point. The guard is one regular
expression against markup nobody promised us, and its failure mode is not a crash —
it is 88 meetings of genuine stewards' comments filed under dates that never raced,
surfacing months later as answers that cite real evidence about a race that did not
happen. A check that only re-runs the guard would agree with the guard, including
when the guard is wrong.

So both checks read the database instead:

**No two meetings share a runner set.** A fallback page is a copy of another
meeting, so contamination shows up as two dates fielding the same horses in the same
races. Real meetings do not do that — a card is 100-odd horses out of a pool of
about 1,200, and even consecutive meetings at the same course share only a handful.

**Every meeting's date matches its own page.** Re-read the archived report the
meeting was built from and ask it again which meeting it is. This re-derives the
answer from the retained page rather than trusting that a check ran at ingest time,
which is what makes it independent — and it is only possible because T10 kept the
pages.

A meeting whose page is missing from the archive is reported, not passed. "Verified"
and "could not check" are the two answers an audit must never blur.

`verify_season` is the third check and the only one that looks outside the database,
holding a season up against the fixture list HKJC published before it ran. It lives
here rather than in `racing_calendar.py` because it is an audit; what it audits just
happens to have an external yardstick.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.orm import Session

from paddock.db.models import Meeting, Race, Runner
from paddock.db.session import session_scope
from paddock.ingest.date_guard import MeetingHeaderMissingError, parse_declared_meeting
from paddock.ingest.dates import season_bounds
from paddock.ingest.pages import latest_page
from paddock.ingest.racing_calendar import PublishedMeeting

# How many announced meetings may be absent before a season stops counting as
# backfilled. T11 set it at three, on the reasoning that HKJC publishes the calendar
# six weeks early and the occasional meeting is abandoned to weather. With the
# published list in hand the check names exactly which dates are missing, so this is
# only the pass/fail line — the list is the part a human acts on.
ABANDONED_TOLERANCE = 3


@dataclass(frozen=True)
class Finding:
    check: str
    """duplicate_runner_set | date_mismatch | unverifiable."""
    race_dates: list[dt.date]
    """The meetings involved. Plural because a duplicate implicates at least two, and
    naming only one of them leaves the operator guessing which to delete."""
    detail: str


@dataclass(frozen=True)
class IntegrityReport:
    findings: list[Finding]
    meetings_checked: int

    @property
    def clean(self) -> bool:
        return not self.findings


def check_integrity(session: Session | None = None) -> IntegrityReport:
    """Audit every stored meeting. Reads only; never repairs.

    Repair is left to a human because the two possible fixes — delete the meeting,
    or re-ingest it — differ in what they do to `incident_comments.id`, and chunks
    cite those ids without a foreign key to protect them.
    """
    if session is not None:
        return _check(session)
    with session_scope() as own_session:
        return _check(own_session)


def _check(session: Session) -> IntegrityReport:
    meetings = list(session.scalars(select(Meeting).order_by(Meeting.race_date)))
    findings = _duplicate_runner_sets(session) + _dates_disagreeing_with_their_page(
        session, meetings
    )
    return IntegrityReport(findings=findings, meetings_checked=len(meetings))


def _duplicate_runner_sets(session: Session) -> list[Finding]:
    """Meetings whose full set of (race number, horse) pairs is identical.

    Grouped in the database rather than in Python: the corpus is ~23,000 runners
    across two seasons, and pulling all of them out to compare sets would be the one
    slow part of an otherwise instant check.
    """
    # Sorted inside the aggregate, so two meetings whose rows were written in a
    # different order still compare equal — the check is about the set, not the walk.
    entry = func.concat(Race.race_no, ":", Runner.horse_id)
    card = (
        select(
            Meeting.race_date.label("race_date"),
            func.array_agg(aggregate_order_by(entry, entry)).label("card"),
        )
        .join(Race, Race.meeting_id == Meeting.id)
        .join(Runner, Runner.race_id == Race.id)
        .group_by(Meeting.id, Meeting.race_date)
        .subquery()
    )

    duplicates = session.execute(
        select(card.c.card, func.array_agg(card.c.race_date))
        .group_by(card.c.card)
        .having(func.count() > 1)
    ).all()

    return [
        Finding(
            check="duplicate_runner_set",
            race_dates=sorted(dates),
            detail=(
                f"{len(dates)} meetings field an identical card of {len(card_entries)} "
                "runners — the shape a served fallback leaves behind"
            ),
        )
        for card_entries, dates in duplicates
    ]


def _dates_disagreeing_with_their_page(session: Session, meetings: list[Meeting]) -> list[Finding]:
    """Ask each meeting's archived report, once more, which meeting it is."""
    findings = []
    for meeting in meetings:
        page = latest_page(session, meeting.source_url) if meeting.source_url else None
        if page is None:
            findings.append(
                Finding(
                    check="unverifiable",
                    race_dates=[meeting.race_date],
                    detail=(
                        f"no archived report page for {meeting.source_url!r} — this meeting "
                        "predates the archive or its pages were deleted, so nothing here "
                        "confirms its date"
                    ),
                )
            )
            continue

        try:
            declared = parse_declared_meeting(page.body)
        except MeetingHeaderMissingError as error:
            findings.append(
                Finding(
                    check="unverifiable",
                    race_dates=[meeting.race_date],
                    detail=f"the archived page carries no meeting header: {error}",
                )
            )
            continue

        if declared != meeting.race_date:
            findings.append(
                Finding(
                    check="date_mismatch",
                    race_dates=[meeting.race_date],
                    detail=(
                        f"stored as {meeting.race_date.isoformat()}, but its own page "
                        f"declares {declared.isoformat()}"
                    ),
                )
            )

    return findings


@dataclass(frozen=True)
class SeasonReport:
    """One season, measured against the fixture list HKJC published before it ran."""

    season: str
    published: int
    """Meetings the sheet says ran — its own printed total, which excludes any it
    marks abandoned."""
    abandoned: int
    ingested: int
    missing: list[dt.date]
    """Announced, and not in the corpus. Usually an abandoned meeting; in bulk, a
    backfill that stopped early."""
    unpublished: list[dt.date]
    """In the corpus, and never announced. A rescheduled meeting looks like this —
    and so does a fallback that got past the guard, which is why it is surfaced."""
    venue_mismatches: list[tuple[dt.date, str, str]]
    """(date, published, stored). Fatal on its own: a meeting that ran at all ran at
    exactly one of two racecourses, so these two sources cannot both be right."""

    @property
    def passed(self) -> bool:
        return not self.venue_mismatches and len(self.missing) <= ABANDONED_TOLERANCE


def verify_season(
    season: str, published: Sequence[PublishedMeeting], session: Session | None = None
) -> SeasonReport:
    """Compare what was ingested for `season` against what HKJC announced.

    The calendar is passed in rather than loaded here so this stays a pure comparison
    — and so the tests can state a two-meeting season instead of all 88.

    Note what this is not: the backfill is driven by generated candidates and the
    guard, neither of which has seen this list. Checking the result against a list
    that produced it would only prove the copy worked.
    """
    if session is not None:
        return _verify(session, season, published)
    with session_scope() as own_session:
        return _verify(own_session, season, published)


def _verify(session: Session, season: str, published: Sequence[PublishedMeeting]) -> SeasonReport:
    start, end = season_bounds(season)
    stored: dict[dt.date, str] = {
        row.race_date: row.racecourse
        for row in session.execute(
            select(Meeting.race_date, Meeting.racecourse).where(
                Meeting.race_date.between(start, end)
            )
        )
    }
    # Two sets, and the difference between them is the abandoned meetings. What the
    # corpus should contain is only what ran; what counts as announced is everything
    # on the sheet, so a meeting abandoned part-way that still published a report is
    # not then reported as a date HKJC never scheduled.
    announced = {meeting.race_date: meeting.racecourse for meeting in published}
    expected = {
        meeting.race_date: meeting.racecourse for meeting in published if not meeting.abandoned
    }

    return SeasonReport(
        season=season,
        published=len(expected),
        abandoned=len(announced) - len(expected),
        ingested=len(stored),
        missing=sorted(expected.keys() - stored.keys()),
        unpublished=sorted(stored.keys() - announced.keys()),
        venue_mismatches=[
            (day, announced[day], stored[day])
            for day in sorted(announced.keys() & stored.keys())
            if announced[day] != stored[day]
        ],
    )
