"""The published fixture lists, and the assumption they disproved.

Three jobs here.

**Guard the transcription.** These files are typed from PDFs and are the yardstick a
whole season gets measured against, so a mis-keyed row would not fail loudly — it
would quietly become the truth. Each sheet prints its own totals (a grand total and
a venue split), and all of them are re-asserted below. That is what turns "typed
carefully" into "checked".

**Prove that Hong Kong does not race on a fixed weekday.** The candidate generator
originally yielded Wednesdays, Saturdays and Sundays. Between them these two seasons
have nine meetings on none of those days, which is why `candidate_dates` now yields
every day — see `test_every_published_meeting_is_reachable`.

**Cross-check HKJC against HKJC.** For 2025-26 there are two independent statements
of the same season: this calendar, and the JSON date index the discovery code
actually uses. They agree exactly, which is worth a test — it validates the index
fixture, the transcription, and `parse_date_list` in one assertion, and it is how
we would find out that either source had moved.
"""

from __future__ import annotations

import collections
import datetime as dt
from pathlib import Path

import pytest

from paddock.ingest.dates import candidate_dates, parse_date_list, season_bounds
from paddock.ingest.racing_calendar import PublishedMeeting, published_meetings

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"

# Season, and the SUMMARY OF FIXTURES the sheet prints for itself. The totals count
# meetings that ran, so 2025-26's 88 excludes the abandoned one.
SEASONS = [
    ("2024-25", 88, {"ST": 51, "HV": 37}),
    ("2025-26", 88, {"ST": 52, "HV": 36}),
]
SEASON_IDS = [season for season, _, _ in SEASONS]


def _calendar(season: str) -> list[PublishedMeeting]:
    meetings = published_meetings(season)
    assert meetings is not None, f"no calendar shipped for {season}"
    return meetings


def _ran(season: str) -> list[PublishedMeeting]:
    """The meetings that actually took place — what the printed totals count."""
    return [meeting for meeting in _calendar(season) if not meeting.abandoned]


# ── The transcriptions check themselves ─────────────────────────────────────────


@pytest.mark.parametrize(("season", "total", "_venues"), SEASONS, ids=SEASON_IDS)
def test_the_calendar_holds_every_meeting_that_ran(
    season: str, total: int, _venues: dict[str, int]
) -> None:
    assert len(_ran(season)) == total, "the sheet's own TOTAL"


@pytest.mark.parametrize(("season", "_total", "venues"), SEASONS, ids=SEASON_IDS)
def test_the_venue_split_matches_the_published_summary(
    season: str, _total: int, venues: dict[str, int]
) -> None:
    """A transposed row keeps the total right and shows up only here."""
    assert collections.Counter(m.racecourse for m in _ran(season)) == venues


@pytest.mark.parametrize("season", SEASON_IDS)
def test_no_date_appears_twice(season: str) -> None:
    dates = [m.race_date for m in _calendar(season)]

    assert len(set(dates)) == len(dates)


@pytest.mark.parametrize("season", SEASON_IDS)
def test_every_meeting_falls_inside_its_own_season(season: str) -> None:
    start, end = season_bounds(season)

    assert all(start <= m.race_date <= end for m in _calendar(season))


def test_a_season_we_never_typed_in_is_absent_not_empty() -> None:
    """ "HKJC planned no meetings" and "we have no calendar" are different facts, and
    only one of them means the corpus is wrong."""
    assert published_meetings("2019-20") is None


# ── An abandoned meeting is a date, not a gap ───────────────────────────────────


def test_the_abandoned_meeting_is_kept_and_marked() -> None:
    """24 September 2025 at Happy Valley never ran. Dropping the row would leave the
    date looking like one nobody ever mentioned, which is the same shape as a meeting
    the backfill lost — so it is kept and flagged instead."""
    abandoned = [m for m in _calendar("2025-26") if m.abandoned]

    assert [m.race_date for m in abandoned] == [dt.date(2025, 9, 24)]
    assert abandoned[0].racecourse == "HV"


def test_the_abandoned_meeting_is_outside_the_printed_total() -> None:
    """89 rows on the sheet, 88 in its own summary. The difference is the reason the
    expected set and the announced set are not the same thing."""
    assert len(_calendar("2025-26")) == 89
    assert len(_ran("2025-26")) == 88


# ── HKJC against HKJC ───────────────────────────────────────────────────────────


def test_the_calendar_and_the_json_index_agree_on_2025_26() -> None:
    """Two independent statements of one season, from the same organisation: the
    fixture sheet a human reads, and the JSON index the discovery code uses.

    One assertion covering the transcription, the index fixture and `parse_date_list`
    at once. If it ever fails, one of those three has moved and the backfill is
    working from something nobody has checked.
    """
    indexed = set(parse_date_list((FIXTURES / "datelist_current_season.json").read_text()))
    published = {m.race_date for m in _ran("2025-26")}

    assert indexed == published


def test_the_index_lists_only_meetings_that_ran() -> None:
    """The abandoned date is absent from the index, so the index is a record of what
    happened rather than a plan of what was intended. That is why an abandoned
    meeting must not count as one the backfill failed to find."""
    indexed = set(parse_date_list((FIXTURES / "datelist_current_season.json").read_text()))

    assert dt.date(2025, 9, 24) not in indexed


# ── What they disprove ──────────────────────────────────────────────────────────


OFF_SCHEDULE = {
    # National Day, Boxing Day, third day of Lunar New Year, HKSAR Establishment Day.
    "2024-25": [
        dt.date(2024, 10, 1),
        dt.date(2024, 12, 26),
        dt.date(2025, 1, 31),
        dt.date(2025, 7, 1),
    ],
    # Note 2025-12-23: a plain Tuesday in December, and not a public holiday — which
    # is what rules out "race weekdays plus holidays" as a cheaper fix.
    "2025-26": [
        dt.date(2025, 10, 30),
        dt.date(2025, 12, 23),
        dt.date(2026, 1, 1),
        dt.date(2026, 2, 19),
        dt.date(2026, 4, 6),
    ],
}


@pytest.mark.parametrize("season", SEASON_IDS)
def test_hong_kong_races_on_days_that_are_not_wednesday_saturday_or_sunday(season: str) -> None:
    off_schedule = sorted(
        m.race_date for m in _ran(season) if m.race_date.weekday() not in {2, 5, 6}
    )

    assert off_schedule == OFF_SCHEDULE[season]


@pytest.mark.parametrize("season", SEASON_IDS)
def test_every_published_meeting_is_reachable(season: str) -> None:
    """The one that matters. A meeting no candidate covers is not rejected by the
    guard — it is never asked about, so it is missing from the corpus with nothing
    anywhere recording that it was skipped.

    This is why `candidate_dates` no longer filters by weekday: nine meetings across
    these two seasons would be invisible, and four of them in 2024-25 alone is
    outside T11's ±3 tolerance before a single request.
    """
    start, end = season_bounds(season)
    candidates = set(candidate_dates(start, end))

    assert [m.race_date for m in _ran(season) if m.race_date not in candidates] == []
