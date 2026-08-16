"""Parser for the legacy LocalResults page — one race per page.

    /racing/information/English/Racing/LocalResults.aspx
        ?RaceDate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N

The modern `en-us/local/information/localresults` page builds its tables in the
browser and serves none of them to a plain fetch, so this legacy page is the source.

Its structure is a header table::

    RACE 1 (634)
    Class 4 - 1200M - (60-40)          Going :  GOOD TO FIRM
    FWD INSURANCE ACT PRIVATE HANDICAP Course : TURF - "A" Course
    HK$ 1,170,000                      Time :   (23.33) (45.62) (1:08.70)

followed by ``table.draggable`` with one row per runner::

    Pla. | Horse No. | Horse | Jockey | Trainer | Act. Wt. | Declar. Horse Wt.
         | Dr. | LBW | Running Position | Finish Time | Win Odds

Unlike the incident report, this endpoint **fails closed**: a date with no meeting
returns a page with no results table, so a missing table is an honest signal and is
raised rather than guarded against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from bs4 import BeautifulSoup, Tag

from paddock.ingest.entities import parse_horse_id
from paddock.ingest.values import (
    as_float,
    as_int,
    parse_finish_time,
    parse_margin,
    parse_running_positions,
    split_name_and_brand,
)

_RACE_NO = re.compile(r"RACE\s*(\d+)", re.I)
_CLASS = re.compile(r"\b(Class\s+\d+|Group\s+(?:One|Two|Three)|Griffin|Restricted)\b", re.I)
_DISTANCE = re.compile(r"\b(\d{3,4})\s*M\b", re.I)
_PRIZE = re.compile(r"HK\$\s*([\d,]+)")
_COURSE = re.compile(r'(TURF|ALL\s*WEATHER)\s*(?:-\s*"([^"]+)"\s*Course)?', re.I)
_DEAD_HEAT = re.compile(r"^(\d+)\s*DH$", re.I)


@dataclass(frozen=True)
class ResultRunner:
    finish_pos: int | None
    dead_heat: bool
    finished: bool
    horse_no: int | None
    horse_name: str
    brand_no: str
    horse_id: str | None
    """HKJC's stable identifier, read from the row's link."""
    jockey: str
    trainer: str
    carried_weight_lb: int | None
    declared_horse_weight_lb: int | None
    draw: int | None
    margin: float | None
    running_positions: list[int]
    finish_time_s: float | None
    win_odds: float | None


@dataclass(frozen=True)
class RaceHeader:
    race_no: int
    name: str | None
    race_class: str | None
    distance_m: int | None
    going: str | None
    track: str | None
    course: str | None
    prize: int | None


@dataclass(frozen=True)
class RaceResults:
    race_no: int
    name: str | None
    race_class: str | None
    distance_m: int | None
    going: str | None
    track: str | None
    course: str | None
    prize: int | None
    runners: list[ResultRunner]


class ResultsParseError(RuntimeError):
    """The page carried no results — usually a date with no meeting."""


def parse_race_results(html: str) -> RaceResults:
    """Parse one race's results page.

    Raises:
        ResultsParseError: no results table, which for this endpoint means the
            meeting does not exist rather than that the markup changed.
    """
    soup = BeautifulSoup(html, "lxml")

    table = _results_table(soup)
    if table is None:
        raise ResultsParseError(
            "no results table on the page — the requested meeting does not exist "
            "(this endpoint returns an empty page rather than an error)"
        )

    runners = [
        runner
        for runner in (_parse_runner(row) for row in table.find_all("tr")[1:])
        if runner is not None
    ]

    header = _parse_header(_header_table(soup))
    return RaceResults(
        race_no=header.race_no,
        name=header.name,
        race_class=header.race_class,
        distance_m=header.distance_m,
        going=header.going,
        track=header.track,
        course=header.course,
        prize=header.prize,
        runners=runners,
    )


def _results_table(soup: BeautifulSoup) -> Tag | None:
    for table in soup.find_all("table"):
        classes = table.get("class") or []
        if "draggable" in classes:
            return cast(Tag, table)
    return None


def _header_table(soup: BeautifulSoup) -> Tag:
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if first_row is not None and _RACE_NO.search(first_row.get_text()):
            return cast(Tag, table)
    raise ResultsParseError("no race header found")


def _parse_header(table: Tag) -> RaceHeader:
    """Read the header from its table rows rather than from flattened text.

    The layout is a two-column grid::

        RACE 1 (634)
        Class 4 - 1200M - (60-40)           Going :  GOOD TO FIRM
        FWD INSURANCE ACT PRIVATE HANDICAP  Course : TURF - "A" Course
        HK$ 1,170,000                       Time :   (23.33) …

    Flattening it first makes the going and the race name indistinguishable — both
    are runs of capitals with no separator — so the row structure is what tells them
    apart. Labels are matched rather than positions, since row order varies.
    """
    rows = [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        for row in table.find_all("tr")
    ]

    race_no: int | None = None
    name: str | None = None
    going: str | None = None
    course_text: str | None = None
    class_distance = ""
    prize: int | None = None

    for cells in rows:
        if not cells:
            continue
        label = cells[1].strip().rstrip(":").strip().lower() if len(cells) > 1 else ""
        value = cells[2].strip() if len(cells) > 2 else ""

        if race_no is None and (match := _RACE_NO.search(cells[0])):
            race_no = int(match.group(1))
        if _CLASS.search(cells[0]) or _DISTANCE.search(cells[0]):
            class_distance = cells[0]
        if label == "going":
            going = value or None
        if label == "course":
            course_text = value or None
            name = cells[0].strip() or None  # the name shares this row
        if prize_match := _PRIZE.search(cells[0]):
            prize = int(prize_match.group(1).replace(",", ""))

    if race_no is None:
        raise ResultsParseError("no race number in header")

    class_match = _CLASS.search(class_distance)
    distance_match = _DISTANCE.search(class_distance)
    course_match = _COURSE.search(course_text or "")

    return RaceHeader(
        race_no=race_no,
        name=re.sub(r"\s+", " ", name).strip() if name else None,
        race_class=class_match.group(1) if class_match else None,
        distance_m=int(distance_match.group(1)) if distance_match else None,
        going=going,
        track=course_match.group(1).upper().replace(" ", "") if course_match else None,
        course=course_match.group(2) if course_match and course_match.group(2) else None,
        prize=prize,
    )


def _parse_runner(row: Tag) -> ResultRunner | None:
    tds = row.find_all("td")
    cells = [cell.get_text(" ", strip=True) for cell in tds]
    if len(cells) < 12:
        return None

    placing, horse_no, horse, jockey, trainer, act_wt, decl_wt, draw, lbw, rp, time, odds = cells[
        :12
    ]
    name, brand_no = split_name_and_brand(horse)
    finish_pos, dead_heat, finished = _parse_placing(placing)

    return ResultRunner(
        finish_pos=finish_pos,
        dead_heat=dead_heat,
        finished=finished,
        horse_no=as_int(horse_no),
        horse_name=name,
        brand_no=brand_no,
        horse_id=_horse_id_from(tds[2]),
        jockey=jockey,
        trainer=trainer,
        carried_weight_lb=as_int(act_wt),
        declared_horse_weight_lb=as_int(decl_wt),
        draw=as_int(draw),
        # A non-finisher has no margin at all; the winner's '---' means zero.
        margin=parse_margin(lbw) if finished else None,
        running_positions=parse_running_positions(rp),
        finish_time_s=parse_finish_time(time),
        win_odds=as_float(odds),
    )


def _horse_id_from(cell: Tag) -> str | None:
    """Read HK_YYYY_Bxxx from the horse cell's link, if it has one."""
    link = cell.find("a")
    if not isinstance(link, Tag):
        return None
    href = link.get("href")
    return parse_horse_id(href) if isinstance(href, str) else None


def _parse_placing(cell: str) -> tuple[int | None, bool, bool]:
    """Same rules as the incident report: plain number, 'N DH', or did not finish."""
    value = cell.strip()
    if value.isdigit():
        return int(value), False, True

    dead_heat = _DEAD_HEAT.match(value)
    if dead_heat is not None:
        return int(dead_heat.group(1)), True, True

    return None, False, False
