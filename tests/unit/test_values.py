"""The shared value conversions in `paddock.ingest.values`."""

from __future__ import annotations

import pytest

from paddock.ingest.values import parse_race_class


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Class 4 - 1200M - (60-40)", "Class 4"),
        # HKJC capitalises the whole word on some pages. Stored as written, it became
        # a second par cell beside "Class 3" with one race in it.
        ("CLASS 3 - 1200M - (80-60)", "Class 3"),
        ("Race:7 (512) THE CHAIRMAN'S SPRINT PRIZE GROUP ONE 1200 m", "Group One"),
        ("Group  Three - 1600M", "Group Three"),
        ("GRIFFIN - 1000M", "Griffin"),
        ("Restricted Race - 1600M", "Restricted"),
        # The four-year-old series names an age where other races name a class.
        # Both pages, as HKJC serves them for the 2025 Classic Mile.
        ("RACE 8 (394) 4 Year Olds - 1600M", "4YO"),
        ("Race:8 (394) THE HONG KONG CLASSIC MILE (Sec1) 4 Year Olds 1600 m", "4YO"),
    ],
)
def test_race_class_is_stored_in_one_spelling(text: str, expected: str) -> None:
    assert parse_race_class(text) == expected


def test_no_class_named_is_none() -> None:
    assert parse_race_class("THE HONG KONG DERBY - 2000M") is None
