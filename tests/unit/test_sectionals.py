"""Parsing per-runner sectional times.

Two things about this endpoint are easy to get wrong and cost real time:

**The date format differs from every other HKJC page.** It wants `DD/MM/YYYY`.
Passing `YYYYMMDD` — the format the results and race-card pages use — returns a page
reading "Information will be released shortly", which looks exactly like a meeting
whose sectionals have not been published yet rather than like a malformed request.

**Each section cell packs three things together**: running position, margin behind
the leader, and the section time, e.g. ``5 2-1/2 22.25 11.00 11.25``. The trailing
numbers are finer splits within the section, so the section time is the *first*
two-decimal number in the cell, not the last.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paddock.ingest.sectionals import parse_sectional_times, sectional_date_param

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


@pytest.fixture(scope="module")
def sectionals() -> object:
    html = (FIXTURES / "sectional_20260426_R1.html").read_text(encoding="utf-8", errors="ignore")
    return parse_sectional_times(html)


def test_date_param_uses_day_first_format() -> None:
    """YYYYMMDD silently returns an empty page — this format is not optional."""
    import datetime as dt

    assert sectional_date_param(dt.date(2026, 4, 26)) == "26/04/2026"


def test_all_runners_present(sectionals) -> None:  # type: ignore[no-untyped-def]
    assert len(sectionals) == 14


def test_section_times_for_the_winner(sectionals) -> None:  # type: ignore[no-untyped-def]
    """A 1200m race has three sections; the finer splits inside them are not sections."""
    winner = next(r for r in sectionals if r.brand_no == "K570")

    assert winner.horse_name == "PACKING KING"
    assert winner.finish_pos == 1
    assert winner.sectional_times == pytest.approx([23.77, 22.25, 22.68])


def test_positions_match_the_results_page(sectionals) -> None:  # type: ignore[no-untyped-def]
    """The same runner's positions appear on both pages; they must agree."""
    winner = next(r for r in sectionals if r.brand_no == "K570")

    assert winner.running_positions == [5, 5, 1]


def test_no_empty_sections_are_recorded(sectionals) -> None:  # type: ignore[no-untyped-def]
    """The table always shows six section columns; a 1200m race only uses three."""
    for runner in sectionals:
        assert runner.sectional_times, "every runner should have at least one section"
        assert all(t > 0 for t in runner.sectional_times)


def test_brand_numbers_are_extracted(sectionals) -> None:  # type: ignore[no-untyped-def]
    """Names on this page carry a non-breaking space before the brand number."""
    for runner in sectionals:
        assert len(runner.brand_no) == 4
        assert "\xa0" not in runner.horse_name
        assert "(" not in runner.horse_name


def test_a_prefixed_brand_number_still_reads_as_a_runner_row() -> None:
    """ "BEAR CHAMP&nbsp;(AJ313)" carries a leading letter that its own link omits.

    The row filter used to require exactly one letter before the digits, so this
    runner was not recognised as a runner row at all — the page parsed, the race
    parsed, and one horse simply had no sectionals. That is the failure mode worth a
    test: it costs data without raising anything.
    """
    html = (FIXTURES / "sectional_20241109_R2_prefixed_brand.html").read_text(
        encoding="utf-8", errors="ignore"
    )

    runners = parse_sectional_times(html)
    bear_champ = next(r for r in runners if r.horse_name == "BEAR CHAMP")

    assert bear_champ.brand_no == "J313"
    assert bear_champ.sectional_times
    assert len(runners) == 14, "every runner in the race, not just the unprefixed ones"
