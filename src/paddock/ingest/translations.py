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
thing that says so — `translate_meeting` refuses a page for a race it did not ask
for.

## The pass over a stored meeting

`translate_meeting` annotates a meeting that is already ingested. It never creates a
race, a runner or a comment, and it must never create a horse, a jockey or a
trainer either: those exist, in English, and this pass is here to give them their
second name. So it counts the three tables before and after itself and reports the
difference, which is expected to be zero and is checked by the caller.

Each Chinese row is paired with the English row for the same `horse_id`, and the
English name that pairing yields is what the jockey and trainer resolvers are keyed
on. Two names for one person arrive together, so neither resolver has to guess.

**A race whose English results page is empty is never asked for in Chinese.** Ten of
a typical card's eleven results pages were captured; a meeting missing one has
nothing to pair a Chinese page against, and fetching it would buy a page that can
only be discarded.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paddock.db.models import Horse, Jockey, Meeting, Race, Trainer
from paddock.db.session import session_scope
from paddock.ingest.entities import resolve_horse, resolve_jockey, resolve_trainer
from paddock.ingest.pages import PageFetcher, fetch_page
from paddock.ingest.pipeline import RESULTS_PATH, record_run, results_params
from paddock.ingest.results import (
    ResultRunner,
    ResultsParseError,
    horse_id_from,
    parse_race_results,
    results_table,
)
from paddock.ingest.values import split_name_and_brand

TRANSLATIONS = "translations"
"""What `ingest_runs.source` records for this pass, beside `incident_report`."""

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


# ── The pass over one meeting ───────────────────────────────────────────────────


class MeetingNotStoredError(RuntimeError):
    """Asked to translate a meeting that was never ingested."""


@dataclass(frozen=True)
class MeetingTranslation:
    """What one meeting's pass did, and what it stepped over."""

    race_date: dt.date
    racecourse: str
    races: int
    """Races whose Chinese names were written."""
    runners: int
    """Rows paired by `horse_id` and named."""
    created: int
    """Horses, jockeys and trainers this pass created. Expected to be zero — every
    one of them is already in the corpus under an English name."""
    races_without_english: list[int] = field(default_factory=list)
    """No English results page, so nothing to pair against. Not fetched in Chinese."""
    races_without_chinese: list[int] = field(default_factory=list)
    """English results exist and the Chinese page carried none."""
    race_number_mismatches: list[int] = field(default_factory=list)
    """The Chinese page named another race. Nothing was written for these."""


def translate_meeting(
    client: PageFetcher,
    race_date: dt.date,
    racecourse: str,
    *,
    refresh: bool = False,
) -> MeetingTranslation:
    """Fill `name_zh` for every horse, jockey and trainer in one stored meeting.

    One transaction for the meeting, as `ingest_meeting` uses: a card half annotated
    is a state nobody has to reason about if it cannot happen.

    Raises:
        MeetingNotStoredError: no such meeting. This pass annotates the corpus and
            never grows it, so an unknown date is a mistake rather than work to do.
    """
    race_numbers = _stored_race_numbers(race_date, racecourse)

    with record_run(TRANSLATIONS, race_date), session_scope() as session:
        return _translate(
            session,
            client,
            race_date=race_date,
            racecourse=racecourse,
            race_numbers=race_numbers,
            refresh=refresh,
        )


def _stored_race_numbers(race_date: dt.date, racecourse: str) -> list[int]:
    """The card, read from the corpus rather than from a page.

    The races are what T11 already stored, so a Chinese page for a race we do not
    have is never requested.
    """
    with session_scope() as session:
        meeting_id = session.scalar(
            select(Meeting.id).where(
                Meeting.race_date == race_date, Meeting.racecourse == racecourse
            )
        )
        if meeting_id is None:
            raise MeetingNotStoredError(
                f"no meeting on {race_date.isoformat()} at {racecourse} — ingest it first"
            )
        return list(
            session.scalars(
                select(Race.race_no).where(Race.meeting_id == meeting_id).order_by(Race.race_no)
            )
        )


def _translate(
    session: Session,
    client: PageFetcher,
    *,
    race_date: dt.date,
    racecourse: str,
    race_numbers: list[int],
    refresh: bool,
) -> MeetingTranslation:
    before = _entity_rows(session)
    without_english: list[int] = []
    without_chinese: list[int] = []
    mismatches: list[int] = []
    races = runners = 0

    for race_no in race_numbers:
        params = results_params(race_date, racecourse, race_no)

        english = _english_runners(client, params, refresh=refresh)
        if english is None:
            without_english.append(race_no)
            continue

        chinese = _chinese_names(client, params, refresh=refresh)
        if chinese is None:
            without_chinese.append(race_no)
            continue

        if chinese.race_no != race_no:
            # Pairing on `horse_id` would already have written nothing, because
            # another race fields other horses. This says so out loud instead.
            mismatches.append(race_no)
            continue

        runners += _name_runners(session, english, chinese.runners, race_date=race_date)
        races += 1

    session.flush()
    return MeetingTranslation(
        race_date=race_date,
        racecourse=racecourse,
        races=races,
        runners=runners,
        created=_entity_rows(session) - before,
        races_without_english=without_english,
        races_without_chinese=without_chinese,
        race_number_mismatches=mismatches,
    )


def _english_runners(
    client: PageFetcher, params: dict[str, str], *, refresh: bool
) -> list[ResultRunner] | None:
    """The English row for each runner, from the archive. T11 fetched these already."""
    page = fetch_page(client, RESULTS_PATH, params, refresh=refresh)
    try:
        return parse_race_results(page.body).runners
    except ResultsParseError:
        return None


def _chinese_names(
    client: PageFetcher, params: dict[str, str], *, refresh: bool
) -> ChineseNames | None:
    """The Chinese row for each runner. This is the one page the pass pays for."""
    page = fetch_page(client, CHINESE_RESULTS_PATH, params, refresh=refresh)
    try:
        return parse_chinese_names(page.body)
    except ResultsParseError:
        return None


def _name_runners(
    session: Session,
    english: list[ResultRunner],
    chinese: list[ChineseRunner],
    *,
    race_date: dt.date,
) -> int:
    """Pair the two languages on `horse_id` and give each entity its second name.

    The English name comes from the paired row rather than from the database, so
    both names reach the resolvers together and neither has to be guessed at. A row
    with no counterpart is skipped: without the English name, resolving a person by
    their Chinese one alone would create them.
    """
    by_horse = {runner.horse_id: runner for runner in english if runner.horse_id}
    named = 0

    for row in chinese:
        counterpart = by_horse.get(row.horse_id) if row.horse_id else None
        if row.horse_id is None or counterpart is None:
            continue

        resolve_horse(
            session,
            horse_id=row.horse_id,
            brand_no=row.brand_no,
            name_zh=row.horse_name,
            seen_on=race_date,
        )
        if counterpart.jockey and row.jockey:
            resolve_jockey(session, name_en=counterpart.jockey, name_zh=row.jockey)
        if counterpart.trainer and row.trainer:
            resolve_trainer(session, name_en=counterpart.trainer, name_zh=row.trainer)
        named += 1

    return named


def _entity_rows(session: Session) -> int:
    """How many horses, jockeys and trainers exist. The pass must not change this."""
    return sum(
        session.scalar(select(func.count()).select_from(model)) or 0
        for model in (Horse, Jockey, Trainer)
    )
