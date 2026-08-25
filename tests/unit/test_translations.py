"""Reading Chinese names off the Chinese results page.

The Chinese page is the same table as the English one with every name in Chinese,
so the parse is deliberately narrow: it reads only what T17a needs — the horse's
identity and the three names — and ignores every result column, which is already
ingested from the English page.

The page carries its own race number as ``第 1 場``. It is parsed and returned so
the caller can refuse a page for a race it did not ask for. This endpoint is new to
the project and no guard has ever been pointed at it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paddock.ingest.results import ResultsParseError
from paddock.ingest.translations import ChineseNames, parse_chinese_names

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8", errors="ignore")


@pytest.fixture(scope="module")
def race1() -> ChineseNames:
    return parse_chinese_names(load("results_20260426_ST_R1_zh.html"))


def test_race_number_is_read_from_the_chinese_header(race1: ChineseNames) -> None:
    """'第 1 場' is the only thing that says which race this page is."""
    assert race1.race_no == 1


def test_every_runner_on_the_card_is_read(race1: ChineseNames) -> None:
    assert len(race1.runners) == 14


def test_names_and_identity_come_off_the_winning_row(race1: ChineseNames) -> None:
    winner = race1.runners[0]

    assert winner.horse_id == "HK_2024_K570"
    assert winner.brand_no == "K570"
    assert winner.horse_name == "包裝天王"
    assert winner.jockey == "潘頓"
    assert winner.trainer == "沈集成"


def test_brand_number_is_stripped_from_every_horse_name(race1: ChineseNames) -> None:
    """The brand is the join key, so it must not be left inside the name."""
    assert all("(" not in runner.horse_name for runner in race1.runners)


def test_every_runner_carries_the_stable_horse_id(race1: ChineseNames) -> None:
    """Without it there is nothing to join a Chinese name to."""
    assert all(runner.horse_id is not None for runner in race1.runners)


def test_a_page_with_no_results_table_raises() -> None:
    """Same contract as the English page: an empty page is an honest 'nothing here'."""
    with pytest.raises(ResultsParseError):
        parse_chinese_names(load("results_20260423_no_meeting.html"))
