"""Race-date discovery.

Two sources, because HKJC only indexes the current season:

**Current season** — a JSON endpoint lists every meeting date authoritatively. This
is why race dates are never hardcoded, and why a rescheduled meeting is picked up
without anyone editing a list.

**Prior seasons** — no index exists, so candidate dates are generated (HK races on
Wednesdays, Saturdays and Sundays) and each is confirmed by fetching it and checking
the page's self-declared meeting date. Roughly one in three candidates is real; the
rest are discarded by the guard.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator

from paddock.ingest.date_guard import MeetingHeaderMissingError, is_genuine
from paddock.ingest.http import HkjcClient

DATE_LIST_PATH = "/racing/information/json/DateList/RacingIncidentReport.aspx"
REPORT_PATH = "/en-us/local/information/racereportfull"

# HK meetings fall on Wednesday, Saturday or Sunday. Generating only these cuts the
# candidate set by more than half before any request is made.
_RACE_WEEKDAYS = frozenset({2, 5, 6})  # Wed, Sat, Sun


def parse_date_list(payload: str) -> list[dt.date]:
    """Parse the meeting-date JSON into dates, newest first."""
    raw = json.loads(payload)["MeetingDateList"]
    return sorted((dt.date.fromisoformat(item.split("T")[0]) for item in raw), reverse=True)


def discover_current_season(client: HkjcClient) -> list[dt.date]:
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


def discover_prior_season(
    client: HkjcClient, start: dt.date, end: dt.date
) -> tuple[list[dt.date], list[dt.date]]:
    """Confirm which candidate dates in a range are real meetings.

    Returns (confirmed, rejected). Rejected dates are returned rather than dropped so
    the caller can log them — an unexpectedly large rejection count is the signal that
    upstream markup changed, not that Hong Kong stopped racing.

    Costs one request per candidate, at the client's rate limit. For a full season
    that is roughly 140 requests.
    """
    confirmed: list[dt.date] = []
    rejected: list[dt.date] = []

    for day in candidate_dates(start, end):
        html = client.get_text(REPORT_PATH, params=report_url_params(day))
        try:
            (confirmed if is_genuine(html, day) else rejected).append(day)
        except MeetingHeaderMissingError:
            # Never silently skip: an absent header means we can no longer tell real
            # pages from substituted ones, which invalidates the whole strategy.
            raise

    return confirmed, rejected
