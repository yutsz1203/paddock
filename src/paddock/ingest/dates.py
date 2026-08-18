"""Race-date discovery.

Two sources, because HKJC only indexes the current season:

**Current season** — a JSON endpoint lists every meeting date authoritatively. This
is why race dates are never hardcoded, and why a rescheduled meeting is picked up
without anyone editing a list.

**Prior seasons** — no index exists, so candidate dates are generated (HK races on
Wednesdays, Saturdays and Sundays) and the guard sorts them out. Roughly one in three
is a real meeting; the rest are rejected.

Confirming a candidate is not done here. It would cost a fetch per date that
ingestion then pays for again, and — because that fetch would go through the client
rather than the page archive — the pages it bought would not be kept. So candidates
are handed to the backfill as they are, and the guard runs inside `ingest_meeting`,
after `fetch_page` has archived the page. A rejected date is therefore free to
re-check on a restart.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterator, Mapping
from typing import Protocol

DATE_LIST_PATH = "/racing/information/json/DateList/RacingIncidentReport.aspx"
REPORT_PATH = "/en-us/local/information/racereportfull"

# HK meetings fall on Wednesday, Saturday or Sunday. Generating only these cuts the
# candidate set by more than half before any request is made.
_RACE_WEEKDAYS = frozenset({2, 5, 6})  # Wed, Sat, Sun


class DateListFetcher(Protocol):
    """The slice of `HkjcClient` date discovery needs.

    Narrower than `PageFetcher`: discovery reads the JSON index and never archives
    anything, because the index is a view of today rather than a page whose content
    an answer might later cite.
    """

    def get_text(self, path: str, params: Mapping[str, str] | None = None) -> str: ...


def parse_date_list(payload: str) -> list[dt.date]:
    """Parse the meeting-date JSON into dates, newest first."""
    raw = json.loads(payload)["MeetingDateList"]
    return sorted((dt.date.fromisoformat(item.split("T")[0]) for item in raw), reverse=True)


def discover_current_season(client: DateListFetcher) -> list[dt.date]:
    """Every meeting date of the current season, from HKJC's own index."""
    return parse_date_list(client.get_text(DATE_LIST_PATH, params={"lang": "en-us"}))


def candidate_dates(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    """Yield every Wed/Sat/Sun in [start, end] — the dates worth checking."""
    day = start
    while day <= end:
        if day.weekday() in _RACE_WEEKDAYS:
            yield day
        day += dt.timedelta(days=1)


def report_url_params(day: dt.date) -> dict[str, str]:
    """The report page takes `date=YYYY/MM/DD`, not the `racedate=YYYYMMDD` used elsewhere."""
    return {"date": day.strftime("%Y/%m/%d")}


# "2025-26" — the two calendar years an HKJC season spans.
_SEASON = re.compile(r"^(\d{4})-(\d{2})$")


def season_bounds(season: str) -> tuple[dt.date, dt.date]:
    """'2025-26' -> the first and last day worth looking for a meeting on.

    September to August rather than the season's actual opening and closing dates,
    which move by a week or two each year. Being generous costs a handful of extra
    candidates and never truncates a season; being exact would silently drop a
    meeting the year HKJC opened in late August.

    Raises:
        ValueError: the string is not two consecutive years — a typo there would
            back-fill the wrong twelve months without saying so.
    """
    match = _SEASON.match(season)
    if match is None:
        raise ValueError(f"{season!r} is not a season in YYYY-YY form, e.g. '2025-26'")

    start_year = int(match.group(1))
    if int(match.group(2)) != (start_year + 1) % 100:
        raise ValueError(
            f"{season!r} is not a season: the years are not consecutive, e.g. '2025-26'"
        )

    return dt.date(start_year, 9, 1), dt.date(start_year + 1, 8, 31)


def dates_for_season(client: DateListFetcher, season: str) -> tuple[list[dt.date], str]:
    """Every date in `season` worth attempting, oldest first, and where they came from.

    One request, whichever branch is taken. If HKJC's index covers the season — it
    only ever covers the current one — those dates are authoritative and complete,
    and nothing is spent on a day that never raced. Otherwise the season is guessed
    at from the race weekdays and the guard rejects the two-in-three that are not
    meetings.

    Returns:
        (dates, source), where source is 'index' or 'candidates'. The caller reports
        it, because "88 dates from the index" and "140 guesses" are different enough
        runs that a log line saying which one this is is worth the tuple.
    """
    start, end = season_bounds(season)
    indexed = sorted(day for day in discover_current_season(client) if start <= day <= end)
    if indexed:
        return indexed, "index"
    return list(candidate_dates(start, end)), "candidates"
