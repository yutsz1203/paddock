"""Parser for the HKJC Racing Incident Report.

One page covers a whole meeting. Its structure is::

    div.race
      div                                   ← one per race
        p.moreClum   "Race:1 (634) NAME (Sec2) Class 4 1200 m"
        table.rirr
          tr  Pla. | Horse No | Colour | Horse | Dr. | Jockey | Incident
          tr  4    | 1        |        | MATZDEN (L133) | 2 | M Zahra | Jumped only fairly…

Three details drive the design:

**"No report." is a sentinel, not a comment.** About a fifth of runners carry that
exact text, meaning the stewards recorded nothing. It is stored as ``comment=None``.
Keeping the literal string would put meaningless sentences in the corpus and let the
agent cite "No report." as evidence about a horse.

**The brand number is in the corpus.** ``MATZDEN (L133)`` yields the join key to
``horses.horse_id``, so comments attach to horses without fuzzy name matching.

**Placings are not always integers.** ``6 DH`` is a dead heat for sixth; ``DNF``,
``WV`` and similar mean the horse did not complete the race. Both parse rather than
raise — the DNF cases carry some of the most informative comments in the corpus,
since a horse that broke down is exactly what a form question needs to surface.

This module is deliberately pure: HTML in, dataclasses out. It performs no I/O and
does not check whether the page is the meeting that was requested — that is
``date_guard``'s job, called by the pipeline before this parser runs.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

# "Race:1 (634) FWD INSURANCE ACT PRIVATE HANDICAP (Sec2) Class 4 1200 m"
_RACE_NO = re.compile(r"Race:\s*(\d+)")
_RACE_CLASS = re.compile(r"\b(Class\s+\d+|Group\s+(?:One|Two|Three)|Griffin|Restricted)\b", re.I)
_DISTANCE = re.compile(r"\b(\d{3,4})\s*m\b", re.I)
_HORSE_NAME_BRAND = re.compile(r"^(.*?)\s*\(([A-Z]\d{3})\)\s*$")

# The stewards' way of saying nothing happened.
_NO_REPORT = "no report"

# "6 DH" — dead heat for sixth.
_DEAD_HEAT = re.compile(r"^(\d+)\s*DH$", re.I)


@dataclass(frozen=True)
class RunnerReport:
    """One runner's line in the incident report."""

    horse_name: str
    brand_no: str
    horse_no: int | None
    draw: int | None
    jockey: str
    finish_pos: int | None
    dead_heat: bool
    finished: bool
    comment: str | None
    """The stewards' narrative, or None when the report said "No report."."""


@dataclass(frozen=True)
class RaceReport:
    race_no: int
    name: str | None
    race_class: str | None
    distance_m: int | None
    runners: list[RunnerReport]


@dataclass(frozen=True)
class MeetingReport:
    race_date: dt.date
    races: list[RaceReport]


class ReportParseError(RuntimeError):
    """The page did not look like an incident report."""


def parse_meeting_report(html: str, race_date: dt.date) -> MeetingReport:
    """Parse a full meeting's incident report.

    Args:
        html: the page body.
        race_date: the meeting date, already validated by `date_guard`. Passed in
            rather than parsed here so this function stays pure and so validation
            cannot be accidentally skipped by calling the parser directly.

    Raises:
        ReportParseError: no race tables were found.
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select("table.rirr")
    if not tables:
        raise ReportParseError("no table.rirr elements — markup may have changed")

    races = [_parse_race(table) for table in tables]
    return MeetingReport(race_date=race_date, races=races)


def _parse_race(table: Tag) -> RaceReport:
    header = _race_header_for(table)
    race_no, name, race_class, distance_m = _parse_race_header(header)

    runners = []
    for row in table.find_all("tr")[1:]:  # skip the column headings
        runner = _parse_runner_row(row)
        if runner is not None:
            runners.append(runner)

    return RaceReport(
        race_no=race_no, name=name, race_class=race_class, distance_m=distance_m, runners=runners
    )


def _race_header_for(table: Tag) -> str:
    """The 'Race:N …' paragraph that precedes the table inside its race div."""
    container = table.parent
    if container is not None:
        heading = container.find("p", class_="moreClum")
        if isinstance(heading, Tag):
            return heading.get_text(" ", strip=True)
    raise ReportParseError("race table had no 'Race:N' heading")


def _parse_race_header(header: str) -> tuple[int, str | None, str | None, int | None]:
    match = _RACE_NO.search(header)
    if match is None:
        raise ReportParseError(f"could not read a race number from {header!r}")
    race_no = int(match.group(1))

    class_match = _RACE_CLASS.search(header)
    distance_match = _DISTANCE.search(header)

    return (
        race_no,
        _race_name(header),
        class_match.group(1) if class_match else None,
        int(distance_match.group(1)) if distance_match else None,
    )


def _race_name(header: str) -> str | None:
    """Strip 'Race:N (id)' from the front and the '(Sec2) Class 4 1200 m' tail."""
    without_prefix = re.sub(r"^Race:\s*\d+\s*(?:\(\d+\))?\s*", "", header).strip()
    name = re.split(r"\s*\(Sec\d+\)|\s*\bClass\s+\d+\b|\s*\bGroup\s+\w+\b", without_prefix)[0]
    cleaned = re.sub(r"\s+", " ", name).strip()
    return cleaned or None


def _parse_runner_row(row: Tag) -> RunnerReport | None:
    cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
    if len(cells) < 7:
        return None  # spacer or malformed row

    placing, horse_no, _colour, horse, draw, jockey, incident = cells[:7]
    name, brand_no = _split_name_and_brand(horse)
    finish_pos, dead_heat, finished = _parse_placing(placing)

    return RunnerReport(
        horse_name=name,
        brand_no=brand_no,
        horse_no=_as_int(horse_no),
        draw=_as_int(draw),
        jockey=jockey,
        finish_pos=finish_pos,
        dead_heat=dead_heat,
        finished=finished,
        comment=_clean_comment(incident),
    )


def _split_name_and_brand(cell: str) -> tuple[str, str]:
    """'MATZDEN (L133)' -> ('MATZDEN', 'L133')."""
    match = _HORSE_NAME_BRAND.match(cell.strip())
    if match is None:
        raise ReportParseError(f"could not read a brand number from {cell!r}")
    return match.group(1).strip(), match.group(2)


def _parse_placing(cell: str) -> tuple[int | None, bool, bool]:
    """Return (finish_pos, dead_heat, finished).

    A plain number is a placing. "6 DH" is a dead heat for that place. Anything else
    ("DNF", "WV", "PU") means the horse did not complete the race, which is recorded
    as no placing rather than discarded — those runners often carry the most
    informative comments in the corpus.
    """
    value = cell.strip()
    if value.isdigit():
        return int(value), False, True

    dead_heat = _DEAD_HEAT.match(value)
    if dead_heat is not None:
        return int(dead_heat.group(1)), True, True

    return None, False, False


def _clean_comment(cell: str) -> str | None:
    """Return the narrative, or None when the stewards recorded nothing."""
    text = re.sub(r"\s+", " ", cell).strip()
    if not text or text.rstrip(".").strip().lower() == _NO_REPORT:
        return None
    return text


def _as_int(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else None
