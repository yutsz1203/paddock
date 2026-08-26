"""What the corpus actually holds.

One question, asked by the demo about itself: which meetings are in here? The UI
prints the answer next to the chat box (T22), because a visitor cannot judge "no
evidence for that" without knowing what the system was ever given.

**Why this is a query and not a constant.** The banner text in T22's acceptance
criteria — "2024-25 and 2025-26 seasons, through 15 July 2026" — is true on the day
it is written and false a week after the live pipeline starts (T14, 6 September).
A demo built around refusing to state what it cannot support should not open with a
claim it stopped checking.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paddock.db.models import Meeting
from paddock.ingest.dates import season_of


@dataclass(frozen=True)
class Coverage:
    """The corpus, as a banner states it."""

    meetings: int
    first_date: dt.date | None
    last_date: dt.date | None
    seasons: list[str]
    """Seasons with at least one meeting, oldest first. Named rather than spanned:
    a gap in the middle is a real gap and the banner must not paper over it."""


def corpus_coverage(session: Session) -> Coverage:
    """Summarise every meeting in the database."""
    meetings = session.scalar(select(func.count()).select_from(Meeting)) or 0
    days = sorted(session.scalars(select(Meeting.race_date).distinct()))

    if not days:
        return Coverage(meetings=0, first_date=None, last_date=None, seasons=[])

    return Coverage(
        meetings=meetings,
        first_date=days[0],
        last_date=days[-1],
        seasons=sorted({season_of(day) for day in days}),
    )
