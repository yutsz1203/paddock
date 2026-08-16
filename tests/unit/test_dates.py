"""Race-date discovery: the JSON index and prior-season candidate generation."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from paddock.ingest.dates import candidate_dates, parse_date_list, report_url_params

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


def test_date_list_parses_a_full_season() -> None:
    dates = parse_date_list((FIXTURES / "datelist_current_season.json").read_text())

    assert len(dates) == 88, "2025-26 had 88 meetings"
    assert dates[0] == dt.date(2026, 7, 15), "newest first"
    assert dates[-1] == dt.date(2025, 9, 7), "season opened 7 September 2025"


def test_date_list_is_sorted_newest_first() -> None:
    dates = parse_date_list((FIXTURES / "datelist_current_season.json").read_text())

    assert dates == sorted(dates, reverse=True)


def test_candidates_are_only_race_weekdays() -> None:
    days = list(candidate_dates(dt.date(2026, 4, 1), dt.date(2026, 4, 30)))

    assert all(d.weekday() in {2, 5, 6} for d in days), "Wed, Sat, Sun only"
    assert dt.date(2026, 4, 26) in days, "a real Sunday meeting must be a candidate"
    assert dt.date(2026, 4, 23) not in days, "Thursday is never a race day"


def test_candidate_range_is_inclusive() -> None:
    # 2026-04-26 is a Sunday, so it qualifies and must appear at both ends.
    days = list(candidate_dates(dt.date(2026, 4, 26), dt.date(2026, 4, 26)))

    assert days == [dt.date(2026, 4, 26)]


def test_report_params_use_the_slashed_date_format() -> None:
    """The report page ignores `racedate=YYYYMMDD` and silently serves the latest meeting."""
    assert report_url_params(dt.date(2026, 4, 26)) == {"date": "2026/04/26"}
