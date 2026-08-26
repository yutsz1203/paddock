"""Every string the demo shows, in both languages.

The toggle is an acceptance criterion (T22), and a half-translated interface is a
worse demo than an English one: a visitor who switches to 中文 and meets three
English labels learns that the bilingual claim is decoration. So the two tables are
the same shape by construction, and these tests check that neither has been left
with an English string in the Chinese column.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re

import pytest

from paddock.ui.text import (
    LANGUAGES,
    data_banner,
    route_name,
    source_kind_name,
    strings,
)

HAN = re.compile(r"[一-鿿]")

FIRST = dt.date(2024, 9, 8)
LAST = dt.date(2026, 7, 15)
SEASONS = ["2024-25", "2025-26"]


# ── The tables ──────────────────────────────────────────────────────────────────


def test_both_languages_are_offered() -> None:
    assert list(LANGUAGES) == ["en", "zh"]


@pytest.mark.parametrize("language", ["en", "zh"])
def test_no_string_is_left_blank(language: str) -> None:
    for value in dataclasses.asdict(strings(language)).values():  # type: ignore[arg-type]
        assert value, f"{language} has an empty string"


def test_the_chinese_table_is_actually_chinese() -> None:
    """A field left as its English default is the failure this catches. `title` is
    the product name and stays as it is, so it is excluded by name and not by luck."""
    table = dataclasses.asdict(strings("zh"))
    for name, value in table.items():
        if name == "title":
            continue
        text = " ".join(value) if isinstance(value, tuple | list) else str(value)
        assert HAN.search(text), f"strings('zh').{name} carries no Chinese"


def test_the_example_questions_are_written_in_each_language() -> None:
    """The examples are one click from an answer, so an English example under the
    Chinese toggle sends a visitor down the English path without meaning to."""
    assert strings("en").examples
    assert not any(HAN.search(example) for example in strings("en").examples)
    assert all(HAN.search(example) for example in strings("zh").examples)


# ── The data-range banner ───────────────────────────────────────────────────────


def test_the_english_banner_states_the_range_plainly() -> None:
    banner = data_banner("en", seasons=SEASONS, last_date=LAST, meetings=176)

    assert banner == "Data: 2024-25 and 2025-26 seasons, through 15 July 2026 — 176 meetings."


def test_the_chinese_banner_states_the_same_range() -> None:
    banner = data_banner("zh", seasons=SEASONS, last_date=LAST, meetings=176)

    assert "2024-25" in banner
    assert "2025-26" in banner
    assert "2026年7月15日" in banner
    assert "176" in banner


def test_one_season_is_not_pluralised() -> None:
    banner = data_banner("en", seasons=["2025-26"], last_date=LAST, meetings=88)

    assert "2025-26 season, through" in banner
    assert "88 meetings" in banner


def test_three_seasons_read_as_a_list() -> None:
    banner = data_banner(
        "en", seasons=["2024-25", "2025-26", "2026-27"], last_date=LAST, meetings=200
    )

    assert banner.startswith("Data: 2024-25, 2025-26 and 2026-27 seasons,")


def test_one_meeting_is_not_pluralised_either() -> None:
    banner = data_banner("en", seasons=["2025-26"], last_date=LAST, meetings=1)

    assert banner.endswith("1 meeting.")


@pytest.mark.parametrize("language", ["en", "zh"])
def test_an_empty_corpus_says_so_rather_than_naming_a_range(language: str) -> None:
    """Falling back to "through today" here would be the exact dishonesty the banner
    exists to prevent."""
    banner = data_banner(language, seasons=[], last_date=None, meetings=0)  # type: ignore[arg-type]

    assert "2026" not in banner
    assert banner == strings(language).banner_empty  # type: ignore[arg-type]


def test_a_single_digit_day_carries_no_leading_zero() -> None:
    banner = data_banner("en", seasons=["2024-25"], last_date=FIRST, meetings=1)

    assert "8 September 2024" in banner


# ── The route label ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("route", ["sql", "vector", "both"])
@pytest.mark.parametrize("language", ["en", "zh"])
def test_every_route_the_api_reports_has_a_label(route: str, language: str) -> None:
    assert route_name(language, route)  # type: ignore[arg-type]


def test_an_unknown_route_falls_back_to_its_own_name() -> None:
    """T16 widens the router. An unlabelled route must show as itself rather than
    blanking the line that says how the answer was reached."""
    assert route_name("en", "fanout") == "fanout"  # type: ignore[arg-type]


# ── The source-card label ───────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["comment", "run"])
@pytest.mark.parametrize("language", ["en", "zh"])
def test_every_kind_the_api_sends_has_a_label(kind: str, language: str) -> None:
    assert source_kind_name(language, kind)  # type: ignore[arg-type]


def test_an_unknown_kind_shows_as_itself() -> None:
    """T15 adds tools, and each one names its rows. An unlabelled kind must still
    identify the card rather than leave it headed by nothing."""
    assert source_kind_name("en", "jockey_stat") == "jockey_stat"  # type: ignore[arg-type]
