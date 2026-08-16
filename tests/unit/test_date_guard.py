"""The guard against HKJC's silent date fallback.

The fixture at the centre of this file is `report_20260423_fallback.html`: a request
for Thursday 23 April 2026, a day with no racing. HKJC answered 200 with a complete,
parseable page for 15 July 2026. Every test here exists so that page can never be
ingested as though it were the 23rd.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from paddock.ingest.date_guard import (
    FallbackDetectedError,
    MeetingHeaderMissingError,
    is_genuine,
    parse_declared_meeting,
    require_genuine,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8", errors="ignore")


def test_declared_date_is_read_from_the_page() -> None:
    html = load("report_20260426_valid.html")

    assert parse_declared_meeting(html) == dt.date(2026, 4, 26)


def test_valid_date_passes() -> None:
    html = load("report_20260426_valid.html")

    assert is_genuine(html, dt.date(2026, 4, 26))


def test_prior_season_date_passes() -> None:
    """Seasons before the current one have no JSON index but are still reachable."""
    html = load("report_20250312_prior_season.html")

    assert is_genuine(html, dt.date(2025, 3, 12))
    assert parse_declared_meeting(html) == dt.date(2025, 3, 12)


def test_fallback_is_rejected() -> None:
    """The core case: a Thursday returns July's meeting, and must not be trusted."""
    html = load("report_20260423_fallback.html")

    assert parse_declared_meeting(html) == dt.date(2026, 7, 15)
    assert not is_genuine(html, dt.date(2026, 4, 23))


def test_require_genuine_raises_on_fallback_with_both_dates() -> None:
    html = load("report_20260423_fallback.html")

    with pytest.raises(FallbackDetectedError) as exc:
        require_genuine(html, dt.date(2026, 4, 23))

    assert exc.value.requested == dt.date(2026, 4, 23)
    assert exc.value.served == dt.date(2026, 7, 15)


def test_require_genuine_is_silent_on_a_real_meeting() -> None:
    require_genuine(load("report_20260426_valid.html"), dt.date(2026, 4, 26))


def test_missing_header_raises_rather_than_guessing() -> None:
    """If HKJC drops the header we lose the ability to detect substitution at all.

    That is a human problem, not a date to skip quietly.
    """
    with pytest.raises(MeetingHeaderMissingError):
        parse_declared_meeting("<html><body>no meeting header here</body></html>")
