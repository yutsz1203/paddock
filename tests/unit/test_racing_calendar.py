"""The published fixture list, and the assumption it disproves.

Two jobs here.

**Guard the transcription.** This file is typed from a PDF and is about to become the
yardstick a whole season is measured against, so a mis-keyed row would not fail
loudly — it would quietly become the truth. The page prints three totals for itself
(88 meetings, 51 Sha Tin, 37 Happy Valley) and all three are re-asserted below, which
is what turns "typed carefully" into "checked".

**Prove that Hong Kong does not race on a fixed weekday.** The candidate generator
originally yielded Wednesdays, Saturdays and Sundays. This calendar says four of the
88 meetings in 2024-25 fell on none of those, which is why `candidate_dates` now
yields every day — see `test_every_published_meeting_is_reachable`.
"""

from __future__ import annotations

import collections
import datetime as dt

from paddock.ingest.dates import candidate_dates, season_bounds
from paddock.ingest.racing_calendar import published_meetings

SEASON = "2024-25"


def _meetings() -> list:
    meetings = published_meetings(SEASON)
    assert meetings is not None
    return meetings


# ── The transcription checks itself ─────────────────────────────────────────────


def test_the_calendar_holds_every_meeting_hkjc_announced() -> None:
    assert len(_meetings()) == 88, "the page's own TOTAL"


def test_the_venue_split_matches_the_published_summary() -> None:
    """40 day + 7 twilight + 4 night at Sha Tin, 1 day + 36 night at Happy Valley.

    A transposed row would keep the total at 88 and show up only here."""
    venues = collections.Counter(m.racecourse for m in _meetings())

    assert venues == {"ST": 51, "HV": 37}


def test_no_date_appears_twice() -> None:
    dates = [m.race_date for m in _meetings()]

    assert len(set(dates)) == len(dates)


def test_every_meeting_falls_inside_its_own_season() -> None:
    start, end = season_bounds(SEASON)

    assert all(start <= m.race_date <= end for m in _meetings())


def test_a_season_we_never_typed_in_is_absent_not_empty() -> None:
    """ "HKJC planned no meetings" and "we have no calendar" are different facts, and
    only one of them means the corpus is wrong."""
    assert published_meetings("2019-20") is None


# ── What it disproves ───────────────────────────────────────────────────────────


def test_hong_kong_races_on_days_that_are_not_wednesday_saturday_or_sunday() -> None:
    """National Day, Boxing Day, the third day of Lunar New Year, and HKSAR
    Establishment Day. All four are public holidays, and all four are real meetings."""
    off_schedule = sorted(
        m.race_date for m in _meetings() if m.race_date.weekday() not in {2, 5, 6}
    )

    assert off_schedule == [
        dt.date(2024, 10, 1),
        dt.date(2024, 12, 26),
        dt.date(2025, 1, 31),
        dt.date(2025, 7, 1),
    ]


def test_every_published_meeting_is_reachable() -> None:
    """The one that matters. A meeting no candidate covers is not rejected by the
    guard — it is never asked about, so it is missing from the corpus with nothing
    anywhere recording that it was skipped.

    This is why `candidate_dates` no longer filters by weekday: four of these 88 would
    be invisible, which is outside T11's own ±3 tolerance before a single request.
    """
    start, end = season_bounds(SEASON)
    candidates = set(candidate_dates(start, end))

    unreachable = sorted(m.race_date for m in _meetings() if m.race_date not in candidates)

    assert unreachable == []
