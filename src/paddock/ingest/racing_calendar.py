"""HKJC's own published fixture list, for the seasons it no longer indexes.

Before a season starts, HKJC publishes a one-page calendar of every meeting it
intends to run, with the racecourse for each. The 2024/25 sheet is dated 26 July
2024 and lists 88 meetings. Once the season is over that page is the only complete
statement of what ran that HKJC still makes; the JSON index only ever covers the
current season, which is the whole reason `dates_for_season` has a second branch.

## What this is for, and what it is deliberately not for

It is **not** the source of dates for a backfill. Driving the backfill from this list
and then checking the backfill against the same list would prove nothing — the same
circularity `integrity.py` avoids by re-deriving dates from archived pages rather
than asking the guard whether the guard worked.

It is the *independent* half of T11's "every accepted date independently verified".
Candidate generation and the guard decide what to ingest, knowing nothing about this
file; afterwards, `paddock check season` holds the result up against what HKJC said
it would run and names every difference.

## It is a plan, not a record

Published six weeks before the season, so a meeting can be abandoned (13 November
2024 lost its last three races to a typhoon signal) or moved. A difference between
this list and the corpus is therefore a thing to look at, not proof of a bug — which
is why the check reports differences rather than failing on them. The exception is a
racecourse that disagrees: a meeting that ran at all ran at exactly one of two
venues, and that is a fact the calendar and the going table cannot both be right
about.

## Transcription

Typed from the published PDF, then checked against the three totals the page prints
for itself: 88 meetings, 51 at Sha Tin (40 day + 7 twilight + 4 night), 37 at Happy
Valley (1 day + 36 night). `test_racing_calendar.py` re-asserts all three, so a
mis-keyed row fails the suite rather than quietly becoming the thing everything else
is measured against.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class PublishedMeeting:
    race_date: dt.date
    racecourse: str


def published_meetings(season: str) -> list[PublishedMeeting] | None:
    """Every meeting HKJC announced for `season`, or None if we have no calendar.

    None rather than an empty list: "HKJC planned no meetings" and "we never typed
    that season's sheet in" are different facts, and only one of them means the
    corpus is wrong.
    """
    source = resources.files("paddock.data").joinpath(f"racing_calendar_{season}.json")
    if not source.is_file():
        return None

    payload = json.loads(source.read_text(encoding="utf-8"))
    return [
        PublishedMeeting(race_date=dt.date.fromisoformat(row["date"]), racecourse=row["racecourse"])
        for row in payload["meetings"]
    ]
