"""Conversions for the value formats HKJC uses across pages.

Shared by the results, sectional and incident-report parsers because the same
formats appear on all of them, and because a margin or a finish time parsed two
different ways would be a silent data inconsistency rather than a visible bug.
"""

from __future__ import annotations

import re

# "Class 4", "Group One", "Griffin". HKJC sometimes capitalises the whole word
# ("CLASS 3"), so the match ignores case and the stored value does not: a GROUP BY
# on race_class must not split one class into two cells.
_RACE_CLASS = re.compile(
    r"\b(Class\s+\d+|Group\s+(?:One|Two|Three)|Griffin|Restricted|4\s*Years?\s*Olds?)\b", re.I
)

# The four-year-old series (Classic Mile, Classic Cup, Derby) names no class at all:
# its header reads "4 Year Olds" where a handicap's reads "Class 4".
_FOUR_YEAR_OLDS = re.compile(r"^4\s*Years?\s*Olds?$", re.I)

# "MATZDEN (L133)", sometimes with a non-breaking space before the bracket, and
# occasionally with a leading letter the horse's own link omits ("BEAR CHAMP (AJ313)"
# for HK_2023_J313). Only the trailing letter+3-digits is the issued brand — see
# `incident_report._HORSE_NAME_BRAND`, which has to agree with this for the join.
_NAME_BRAND = re.compile(r"^(.*?)[\s\xa0]*\(([A-Z]?)([A-Z]\d{3})\)\s*$")

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
    return match.group(1).strip(), match.group(3)


def parse_race_class(text: str) -> str | None:
    """'CLASS 3' -> 'Class 3', '4 Year Olds' -> '4YO'. None when no class is named."""
    match = _RACE_CLASS.search(text)
    if match is None:
        return None
    value = " ".join(match.group(1).split())
    if _FOUR_YEAR_OLDS.match(value):
        return "4YO"
    return value.title()


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
