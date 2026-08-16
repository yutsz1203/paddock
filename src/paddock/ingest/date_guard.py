"""Detects when HKJC serves a meeting other than the one requested.

## Why this module exists

`racereportfull?date=YYYY/MM/DD` returns HTTP 200 with a complete, well-formed page
for *any* date — including dates on which no racing took place. It does not error;
it silently serves the most recent meeting instead. Requesting a Thursday in April
returns July's meeting, parsed perfectly, indistinguishable from a real result.

Ingesting that would write real data under wrong dates. Every downstream answer
would cite a genuine comment attached to a race that never happened, and the failure
would surface as inexplicable retrieval bugs weeks later, long after the cause.

## How it detects the substitution

The page states which meeting it is showing, in a header of the form::

    Race Meeting: 26/04/2026 (Sun)

So the guard parses that self-declared date and compares it to the date requested.
A mismatch means the server fell back, and the page is discarded.

This is deliberately simpler than the cross-source check the plan originally called
for (comparing parsed runners against the results page). The page already tells us
what it is; asking it directly needs no second request and no dependency on any
other parser. If HKJC ever removes the header, `parse_declared_meeting` raises
rather than guessing, and the fallback to a cross-source check is a known option.
"""

from __future__ import annotations

import datetime as dt
import re

from bs4 import BeautifulSoup

# "Race Meeting: 26/04/2026 (Sun)" — the venue sometimes follows, so this is not anchored.
_MEETING_HEADER = re.compile(r"Race\s*Meeting:\s*(\d{2})/(\d{2})/(\d{4})", re.IGNORECASE)


class MeetingHeaderMissingError(RuntimeError):
    """The page carried no 'Race Meeting:' header — markup changed upstream."""


class FallbackDetectedError(RuntimeError):
    """The server served a different meeting from the one requested."""

    def __init__(self, requested: dt.date, served: dt.date) -> None:
        super().__init__(
            f"HKJC served {served.isoformat()} for a request for {requested.isoformat()} — "
            "the requested date has no meeting, and the page was discarded."
        )
        self.requested = requested
        self.served = served


def parse_declared_meeting(html: str) -> dt.date:
    """Return the meeting date the page says it is showing.

    Raises:
        MeetingHeaderMissingError: the header is absent, so the page cannot be trusted.
    """
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    match = _MEETING_HEADER.search(text)
    if match is None:
        raise MeetingHeaderMissingError(
            "no 'Race Meeting: DD/MM/YYYY' header found; HKJC markup may have changed"
        )

    day, month, year = (int(g) for g in match.groups())
    return dt.date(year, month, day)


def is_genuine(html: str, requested: dt.date) -> bool:
    """True when the page really is the meeting that was asked for.

    Returns False on a fallback. Propagates `MeetingHeaderMissingError`, because a missing
    header is an upstream change that needs a human, not a date to skip quietly.
    """
    return parse_declared_meeting(html) == requested


def require_genuine(html: str, requested: dt.date) -> None:
    """Raise `FallbackDetectedError` unless the page is the requested meeting.

    Ingestion calls this before parsing anything. Failing loudly here is the whole
    point: a skipped date is recoverable, a silently mislabelled meeting is not.
    """
    served = parse_declared_meeting(html)
    if served != requested:
        raise FallbackDetectedError(requested=requested, served=served)
