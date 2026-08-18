"""Race-date discovery: the JSON index and prior-season candidate generation."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from paddock.ingest.dates import (
    DATE_LIST_PATH,
    candidate_dates,
    dates_for_season,
    parse_date_list,
    report_url_params,
    season_bounds,
)

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


# ── Seasons ─────────────────────────────────────────────────────────────────────


def test_a_season_runs_september_to_august() -> None:
    """HKJC's season opens in early September and closes in July. The bounds are
    generous at both ends so a late opener or a July finale still falls inside."""
    assert season_bounds("2025-26") == (dt.date(2025, 9, 1), dt.date(2026, 8, 31))


def test_the_prior_season_is_the_one_before_it() -> None:
    assert season_bounds("2024-25") == (dt.date(2024, 9, 1), dt.date(2025, 8, 31))


@pytest.mark.parametrize("bad", ["2024-2025", "2024-26", "24-25", "2024", "next"])
def test_a_season_that_is_not_two_consecutive_years_is_rejected(bad: str) -> None:
    """A typo here would silently backfill the wrong twelve months."""
    with pytest.raises(ValueError, match="season"):
        season_bounds(bad)


def test_the_index_supplies_the_season_it_covers() -> None:
    """88 real dates beat 140 guesses: no request is spent on a date that never raced."""
    client = _IndexOnly((FIXTURES / "datelist_current_season.json").read_text())

    dates, source = dates_for_season(client, "2025-26")

    assert source == "index"
    assert len(dates) == 88
    assert dates[0] == dt.date(2025, 9, 7), "oldest first — a backfill walks forward"


def test_a_season_the_index_does_not_cover_falls_back_to_candidates() -> None:
    """HKJC indexes only the current season, so 2024-25 has to be guessed at and
    filtered by the guard during ingestion."""
    client = _IndexOnly((FIXTURES / "datelist_current_season.json").read_text())

    dates, source = dates_for_season(client, "2024-25")

    assert source == "candidates"
    assert all(d.weekday() in {2, 5, 6} for d in dates)
    assert dt.date(2025, 3, 12) in dates, "a real Wednesday meeting must be a candidate"
    assert len(dates) > 100, "roughly one candidate in three is a meeting"


class _IndexOnly:
    """Answers the date-list request and nothing else."""

    def __init__(self, payload: str) -> None:
        self.payload = payload

    def get_text(self, path: str, params: object = None) -> str:
        assert path == DATE_LIST_PATH, f"unexpected request for {path}"
        return self.payload
