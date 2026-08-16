"""Conversions for the value formats HKJC uses across pages.

Shared by the results and sectional parsers because the same formats appear on
both, and because a margin or a finish time parsed two different ways would be a
silent data inconsistency rather than a visible bug.
"""

from __future__ import annotations

import re

# "MATZDEN (L133)", sometimes with a non-breaking space before the bracket.
_NAME_BRAND = re.compile(r"^(.*?)[\s\xa0]*\(([A-Z]\d{3})\)\s*$")

# "1:08.70" (minutes) or "58.42" (seconds only)
_MINUTES_SECONDS = re.compile(r"^(\d+):(\d{2}(?:\.\d+)?)$")

# "4-1/2" lengths, "3", "2-3/4"
_WHOLE_AND_FRACTION = re.compile(r"^(\d+)-(\d+)/(\d+)$")
_FRACTION_ONLY = re.compile(r"^(\d+)/(\d+)$")

# Short margins have names rather than numbers. These are the conventional
# length equivalents used in racing form.
_NAMED_MARGINS = {
    "N": 0.05,  # nose
    "SH": 0.1,  # short head
    "HD": 0.2,  # head
    "SN": 0.05,  # short neck
    "NK": 0.3,  # neck
}

# The winner, and any field with no value, is written as dashes.
_ABSENT = {"", "---", "--", "-", "N/A"}


def split_name_and_brand(cell: str) -> tuple[str, str]:
    """'PACKING KING (K570)' -> ('PACKING KING', 'K570')."""
    match = _NAME_BRAND.match(cell.replace("\xa0", " ").strip())
    if match is None:
        raise ValueError(f"no brand number in {cell!r}")
    return match.group(1).strip(), match.group(2)


def parse_finish_time(cell: str) -> float | None:
    """'1:08.70' -> 68.70 seconds. Returns None when absent (e.g. a non-finisher)."""
    value = cell.strip()
    if value in _ABSENT:
        return None

    match = _MINUTES_SECONDS.match(value)
    if match is not None:
        return int(match.group(1)) * 60 + float(match.group(2))

    try:
        return float(value)
    except ValueError:
        return None


def parse_margin(cell: str) -> float | None:
    """Lengths behind the leader.

    The winner's cell is '---', which means zero lengths behind — a fact, not a
    missing value, so it returns 0.0. A non-finisher has no margin at all and
    returns None; the caller distinguishes them by whether the horse finished.
    """
    value = cell.strip().upper()
    if value in _ABSENT:
        return 0.0

    if value in _NAMED_MARGINS:
        return _NAMED_MARGINS[value]

    mixed = _WHOLE_AND_FRACTION.match(value)
    if mixed is not None:
        whole, numerator, denominator = (int(g) for g in mixed.groups())
        return whole + numerator / denominator

    fraction = _FRACTION_ONLY.match(value)
    if fraction is not None:
        return int(fraction.group(1)) / int(fraction.group(2))

    try:
        return float(value)
    except ValueError:
        return None


def parse_running_positions(cell: str) -> list[int]:
    """'5 5 1' -> [5, 5, 1]. A non-finisher may have fewer positions than sections."""
    return [int(token) for token in cell.replace("\xa0", " ").split() if token.isdigit()]


def as_int(cell: str) -> int | None:
    value = cell.replace(",", "").strip()
    return int(value) if value.isdigit() else None


def as_float(cell: str) -> float | None:
    value = cell.replace(",", "").strip()
    if value in _ABSENT:
        return None
    try:
        return float(value)
    except ValueError:
        return None
