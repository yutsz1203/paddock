"""Parsing race results from the legacy LocalResults page.

The modern `en-us/local/information/localresults` page renders client-side and
serves no tables to a plain HTTP fetch. The legacy ASPX page returns the whole race
server-side, so that is what we ingest.

Unlike the incident report, this endpoint **fails closed**: a date with no meeting
returns a page with no results table rather than silently substituting another
meeting. That asymmetry is why `test_absent_meeting_raises` asserts an exception
instead of reaching for the date guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paddock.ingest.results import ResultsParseError, parse_race_results

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8", errors="ignore")


@pytest.fixture(scope="module")
def race1() -> object:
    return parse_race_results(load("results_20260426_ST_R1.html"))


@pytest.fixture(scope="module")
def dead_heat() -> object:
    return parse_race_results(load("results_20250312_HV_R7_deadheat.html"))


@pytest.fixture(scope="module")
def dnf() -> object:
    return parse_race_results(load("results_20250312_HV_R5_dnf.html"))


# ── Race header ─────────────────────────────────────────────────────────────────


def test_race_header(race1) -> None:  # type: ignore[no-untyped-def]
    assert race1.race_no == 1
    assert race1.name == "FWD INSURANCE ACT PRIVATE HANDICAP"
    assert race1.race_class == "Class 4"
    assert race1.distance_m == 1200
    assert race1.going == "GOOD TO FIRM"
    assert race1.track == "TURF"
    assert race1.course == "A"
    assert race1.prize == 1_170_000


def test_header_of_a_longer_race(dnf) -> None:  # type: ignore[no-untyped-def]
    assert dnf.race_no == 5
    assert dnf.distance_m == 1650
    assert dnf.name == "THE IRELAND TROPHY (HANDICAP)"


# ── Runner fields ───────────────────────────────────────────────────────────────


def test_full_field_is_parsed(race1) -> None:  # type: ignore[no-untyped-def]
    assert len(race1.runners) == 14


def test_winner_fields(race1) -> None:  # type: ignore[no-untyped-def]
    winner = race1.runners[0]

    assert winner.finish_pos == 1
    assert winner.horse_name == "PACKING KING"
    assert winner.brand_no == "K570"
    assert winner.horse_no == 13
    assert winner.jockey == "Z Purton"
    assert winner.trainer == "C S Shum"
    assert winner.carried_weight_lb == 122
    assert winner.declared_horse_weight_lb == 1133
    assert winner.draw == 8
    assert winner.win_odds == pytest.approx(3.3)


def test_finish_time_is_converted_to_seconds(race1) -> None:  # type: ignore[no-untyped-def]
    """'1:08.70' is 68.70 seconds — stored as a number so it can be compared."""
    assert race1.runners[0].finish_time_s == pytest.approx(68.70)


def test_winner_margin_is_zero_not_none(race1) -> None:  # type: ignore[no-untyped-def]
    """The winner's LBW is '---'. Zero lengths behind is a fact, not missing data."""
    assert race1.runners[0].margin == pytest.approx(0.0)


def test_running_positions_are_parsed(race1) -> None:  # type: ignore[no-untyped-def]
    """'5 5 1' is the horse's position at each section."""
    assert race1.runners[0].running_positions == [5, 5, 1]


def test_fractional_margin(dead_heat) -> None:  # type: ignore[no-untyped-def]
    """'4-1/2' lengths must become 4.5, not 4 and not a crash."""
    eighth = next(r for r in dead_heat.runners if r.brand_no == "G264")

    assert eighth.margin == pytest.approx(4.5)


# ── Dead heats and non-finishers ────────────────────────────────────────────────


def test_dead_heat_shares_a_position(dead_heat) -> None:  # type: ignore[no-untyped-def]
    tied = [r for r in dead_heat.runners if r.dead_heat]

    assert len(tied) == 2
    assert {r.brand_no for r in tied} == {"J082", "K080"}
    assert all(r.finish_pos == 6 for r in tied)
    assert all(r.finish_time_s == pytest.approx(69.49) for r in tied)


def test_dnf_keeps_the_runner_but_has_no_result(dnf) -> None:  # type: ignore[no-untyped-def]
    """A horse that did not finish still ran — and its comment is often the point."""
    runner = next(r for r in dnf.runners if r.brand_no == "G144")

    assert runner.horse_name == "INTREPID WINNER"
    assert runner.finished is False
    assert runner.finish_pos is None
    assert runner.finish_time_s is None
    assert runner.margin is None
    assert runner.win_odds == pytest.approx(38.0)
    assert runner.running_positions == [6], "it was 6th when it went amiss"


def test_dnf_race_keeps_the_whole_field(dnf) -> None:  # type: ignore[no-untyped-def]
    assert len(dnf.runners) == 12


# ── Failure modes ───────────────────────────────────────────────────────────────


def test_absent_meeting_raises(race1) -> None:  # type: ignore[no-untyped-def]
    """This endpoint fails closed — no meeting means no results table.

    Contrast with the incident report, which fails open and needs `date_guard`.
    """
    with pytest.raises(ResultsParseError):
        parse_race_results(load("results_20260423_no_meeting.html"))


def test_every_runner_can_be_joined(race1) -> None:  # type: ignore[no-untyped-def]
    for runner in race1.runners:
        assert runner.brand_no
        assert runner.horse_name
        assert "(" not in runner.horse_name


# ── Horse identity ──────────────────────────────────────────────────────────────


def test_horse_id_is_read_from_the_row_link(race1) -> None:  # type: ignore[no-untyped-def]
    winner = race1.runners[0]

    assert winner.horse_id == "HK_2024_K570"
    assert winner.horse_id.endswith(winner.brand_no)


def test_every_runner_carries_a_horse_id(race1) -> None:  # type: ignore[no-untyped-def]
    for runner in race1.runners:
        assert runner.horse_id is not None
        assert runner.horse_id.endswith(runner.brand_no)
