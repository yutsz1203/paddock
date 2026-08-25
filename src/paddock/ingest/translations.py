"""Reading Chinese names off HKJC's Chinese results page.

    /racing/information/Chinese/Racing/LocalResults.aspx
        ?RaceDate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N

The same page as `results.py` reads, in the other language. Every result column is
identical, so this parser reads none of them: the results are already ingested from
the English page, and reading them twice would only create a second opinion about
the same race. What it reads is the four things T17a needs::

    名次  馬號  馬名              騎師   練馬師
    1     13   包裝天王 (K570)   潘頓   沈集成

The horse's link carries `horseid=HK_2024_K570` exactly as the English page does, so
a Chinese name joins to a horse by identity rather than by row position or by any
match between the two names.

**The race number is parsed and returned rather than assumed.** This URL is new to
the project, and no guard has been pointed at it. If HKJC substitutes another race
the way its report endpoint substitutes another meeting, the header is the cheapest
thing that says so — see `translate_race`, which refuses a page for a race it did
not ask for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from paddock.ingest.results import ResultsParseError, horse_id_from, results_table
from paddock.ingest.values import split_name_and_brand

CHINESE_RESULTS_PATH = "/racing/information/Chinese/Racing/LocalResults.aspx"

# "第 1 場 (634)" — the Chinese header's race number.
_RACE_NO_ZH = re.compile(r"第\s*(\d+)\s*場")

# Column order matches the English page exactly, and only these three are read.
_HORSE = 2
_JOCKEY = 3
_TRAINER = 4


@dataclass(frozen=True)
class ChineseRunner:
    """One row's identity and its three Chinese names."""

    horse_id: str | None
    """HKJC's stable identifier, read from the row's link — the join key."""
    brand_no: str
    horse_name: str
    jockey: str
    trainer: str


@dataclass(frozen=True)
class ChineseNames:
    race_no: int
    runners: list[ChineseRunner]


def parse_chinese_names(html: str) -> ChineseNames:
    """Parse one race's Chinese results page for names only.

    Raises:
        ResultsParseError: no results table, or no race number in the header. Same
            contract as the English page: an empty page means the meeting does not
            exist rather than that the markup moved.
    """
    soup = BeautifulSoup(html, "lxml")

    table = results_table(soup)
    if table is None:
        raise ResultsParseError(
            "no results table on the Chinese page — the requested race does not exist"
        )

    runners = [
        runner
        for runner in (_parse_runner(row) for row in table.find_all("tr")[1:])
        if runner is not None
    ]
    return ChineseNames(race_no=_race_no(soup), runners=runners)


def _race_no(soup: BeautifulSoup) -> int:
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if first_row is not None and (match := _RACE_NO_ZH.search(first_row.get_text())):
            return int(match.group(1))
    raise ResultsParseError("no race number in the Chinese header")


def _parse_runner(row: Tag) -> ChineseRunner | None:
    cells = row.find_all("td")
    if len(cells) < 12:
        return None

    text = [cell.get_text(" ", strip=True) for cell in cells]
    name, brand_no = split_name_and_brand(text[_HORSE])

    return ChineseRunner(
        horse_id=horse_id_from(cells[_HORSE]),
        brand_no=brand_no,
        horse_name=name,
        jockey=text[_JOCKEY],
        trainer=text[_TRAINER],
    )
