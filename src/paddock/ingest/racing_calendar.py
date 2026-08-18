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

Published before the season, so a meeting can be abandoned or moved. HKJC reissues
the sheet when that happens — the 2025/26 file here is the amendment as at 4
February 2026 — but a difference between it and the corpus is still a thing to look
at rather than proof of a bug, which is why the check reports differences instead of
failing on them. The exception is a racecourse that disagrees: a meeting that ran at
all ran at exactly one of two venues, and that is a fact the calendar and the going
table cannot both be right about.

## Abandoned meetings are kept, and marked

The 2025/26 sheet lists 89 dates and totals 88, because 24 September 2025 at Happy
Valley was abandoned. HKJC's own JSON index omits that date entirely, which says the
index is a record of what ran rather than a plan of what was intended.

The row is kept rather than deleted, because a deleted row and a date nobody ever
mentioned are indistinguishable — and "the backfill lost a meeting" is exactly the
shape this file exists to detect. Marked, it is excluded from what the corpus is
expected to contain while still counting as announced, so a meeting that was
abandoned part-way and published a partial report is not then reported as one HKJC
never scheduled.

## Transcription

Typed from the published PDFs, then checked against the totals each sheet prints for
itself — 88 meetings for both seasons, split 51/37 in 2024-25 and 52/36 in 2025-26.
`test_racing_calendar.py` re-asserts all of them, so a mis-keyed row fails the suite
rather than quietly becoming the thing everything else is measured against. For
2025-26 there is a stronger check available and it is used: the transcription is
compared date for date against HKJC's own JSON index, and they agree exactly.
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
    abandoned: bool = False
    """Announced, and did not run. Excluded from what the corpus should contain, but
    still announced — see the module docstring."""


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
        PublishedMeeting(
            race_date=dt.date.fromisoformat(row["date"]),
            racecourse=row["racecourse"],
            abandoned=row.get("abandoned", False),
        )
        for row in payload["meetings"]
    ]
