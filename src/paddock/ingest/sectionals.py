"""Parser for per-runner sectional times.

    /en-us/local/information/displaysectionaltime?racedate=DD/MM/YYYY&RaceNo=N

**The date format is day-first here and nowhere else.** Every other HKJC endpoint we
use takes `YYYYMMDD` or `YYYY/MM/DD`. Passing those to this page returns a valid-looking
page reading "Information will be released shortly", which is indistinguishable from a
meeting whose sectionals genuinely have not been published. `sectional_date_param`
exists so that format is stated once rather than remembered at each call site.

Each section cell packs three values together::

    5 2-1/2 22.25 11.00 11.25
    │ │     │     └──────────┴─ finer splits *within* the section
    │ │     └─ the section time we want
    │ └─ margin behind the leader
    └─ running position at that point

So the section time is the **first** two-decimal number in the cell, not the last.
Taking the last would silently record 200m splits as full section times, which would
look plausible and be wrong by half.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from paddock.ingest.values import as_int, split_name_and_brand

_SECTION_TIME = re.compile(r"\d+\.\d{2}")
_LEADING_POSITION = re.compile(r"^\s*(\d+)")


@dataclass(frozen=True)
class RunnerSectionals:
    finish_pos: int | None
    horse_no: int | None
    horse_name: str
    brand_no: str
    sectional_times: list[float]
    running_positions: list[int]


class SectionalParseError(RuntimeError):
    """The page carried no sectional table."""


def sectional_date_param(day: dt.date) -> str:
    """This endpoint wants DD/MM/YYYY. Other HKJC pages do not."""
    return day.strftime("%d/%m/%Y")


def parse_sectional_times(html: str) -> list[RunnerSectionals]:
    """Parse every runner's sectional times and positions for one race."""
    soup = BeautifulSoup(html, "lxml")

    rows = _runner_rows(soup)
    if not rows:
        raise SectionalParseError(
            "no sectional rows found — sectionals may not be published yet, or the "
            "date was not in DD/MM/YYYY format"
        )

    return [runner for runner in (_parse_row(row) for row in rows) if runner is not None]


def _runner_rows(soup: BeautifulSoup) -> list[Tag]:
    """Rows carrying a horse name with a brand number are the runner rows."""
    rows = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        if re.search(r"\([A-Z]\d{3}\)", cells[2].get_text(" ", strip=True)):
            rows.append(row)
    return rows


def _parse_row(row: Tag) -> RunnerSectionals | None:
    cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
    if len(cells) < 4:
        return None

    try:
        name, brand_no = split_name_and_brand(cells[2])
    except ValueError:
        return None

    times: list[float] = []
    positions: list[int] = []

    # Columns 0-2 are placing, horse number and horse; 3 onwards are the sections,
    # of which a race uses only as many as its distance requires. The table always
    # renders six, so trailing empties are skipped rather than recorded as zeros.
    for cell in cells[3:]:
        text = cell.replace("\xa0", " ").strip()
        if not text:
            continue

        # The final column is the overall finish time ("1:08.70"). Left in, its
        # "08.70" would be recorded as a fourth section — plausible-looking and wrong.
        if ":" in text:
            continue

        time_match = _SECTION_TIME.search(text)
        if time_match is None:
            continue
        times.append(float(time_match.group()))

        position_match = _LEADING_POSITION.match(text)
        if position_match is not None:
            positions.append(int(position_match.group(1)))

    return RunnerSectionals(
        finish_pos=as_int(cells[0]),
        horse_no=as_int(cells[1]),
        horse_name=name,
        brand_no=brand_no,
        sectional_times=times,
        running_positions=positions,
    )
