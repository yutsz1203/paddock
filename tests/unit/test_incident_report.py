"""Parsing the Racing Incident Report.

Fixtures are real pages from both seasons, so these tests fail if HKJC's markup
assumptions were wrong — and, because CI never fetches from HKJC, a failure here
always means our code changed rather than their site.

The assertion that matters most is `test_no_report_becomes_none`. Roughly a fifth of
runners carry the literal text "No report.", which means the stewards saw nothing
worth recording. Storing that string as a comment would put 28 sentences of
meaningless text into the corpus per meeting and let the agent cite "No report." as
evidence about a horse. Absence has to stay absence.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from paddock.ingest.incident_report import parse_meeting_report

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8", errors="ignore")


@pytest.fixture(scope="module")
def current_season() -> object:
    return parse_meeting_report(load("report_20260426_valid.html"), dt.date(2026, 4, 26))


@pytest.fixture(scope="module")
def prior_season() -> object:
    return parse_meeting_report(load("report_20250312_prior_season.html"), dt.date(2025, 3, 12))


# ── Structure ───────────────────────────────────────────────────────────────────


def test_all_races_are_found(current_season) -> None:  # type: ignore[no-untyped-def]
    assert len(current_season.races) == 11
    assert [r.race_no for r in current_season.races] == list(range(1, 12))


def test_prior_season_markup_also_parses(prior_season) -> None:  # type: ignore[no-untyped-def]
    """2024-25 pages must parse with the same code — no per-season branching."""
    assert len(prior_season.races) == 9
    assert [r.race_no for r in prior_season.races] == list(range(1, 10))


def test_meeting_date_is_carried_through(current_season) -> None:  # type: ignore[no-untyped-def]
    assert current_season.race_date == dt.date(2026, 4, 26)


def test_race_metadata_is_extracted(current_season) -> None:  # type: ignore[no-untyped-def]
    race = current_season.races[0]

    assert race.name == "FWD INSURANCE ACT PRIVATE HANDICAP"
    assert race.race_class == "Class 4"
    assert race.distance_m == 1200


# ── Row counts ──────────────────────────────────────────────────────────────────


def test_exact_runner_count(current_season) -> None:  # type: ignore[no-untyped-def]
    assert sum(len(r.runners) for r in current_season.races) == 142


def test_exact_commented_count(current_season) -> None:  # type: ignore[no-untyped-def]
    """114 of 142 runners drew a comment; the other 28 said 'No report.'"""
    commented = [run for r in current_season.races for run in r.runners if run.comment is not None]

    assert len(commented) == 114


def test_prior_season_counts(prior_season) -> None:  # type: ignore[no-untyped-def]
    runners = [run for r in prior_season.races for run in r.runners]

    assert len(runners) == 106
    assert sum(1 for run in runners if run.comment is not None) == 91


# ── Field extraction ────────────────────────────────────────────────────────────


def test_runner_fields(current_season) -> None:  # type: ignore[no-untyped-def]
    runner = current_season.races[0].runners[0]

    assert runner.horse_name == "MATZDEN"
    assert runner.brand_no == "L133"
    assert runner.finish_pos == 4
    assert runner.horse_no == 1
    assert runner.draw == 2
    assert runner.jockey == "M Zahra"
    assert runner.comment is not None
    assert runner.comment.startswith("Jumped only fairly.")
    assert "PACKING KING" in runner.comment


def test_brand_number_is_stripped_from_the_name(current_season) -> None:
    """The brand number is the join key to horse_id, so it must not stay in the name."""
    for race in current_season.races:
        for runner in race.runners:
            assert "(" not in runner.horse_name
            assert len(runner.brand_no) == 4


# ── Absence, dead heats, non-finishers ──────────────────────────────────────────


def test_no_report_becomes_none(current_season) -> None:  # type: ignore[no-untyped-def]
    """'No report.' is a sentinel for a clean run, not a comment about the horse."""
    comments = [run.comment for r in current_season.races for run in r.runners]

    assert None in comments, "runners with no incident must be present with comment=None"
    assert not any(c and "No report" in c for c in comments)


def test_runner_without_a_comment_is_still_recorded(current_season) -> None:  # type: ignore[no-untyped-def]
    """Absence is data. The runner exists; only the comment is missing."""
    clean = next(run for r in current_season.races for run in r.runners if run.comment is None)

    assert clean.horse_name
    assert clean.brand_no
    assert clean.finish_pos is not None


def test_dead_heat_parses_to_its_position(prior_season) -> None:  # type: ignore[no-untyped-def]
    """'6 DH' means dead heat for 6th — a position, not an unparseable string."""
    race7 = next(r for r in prior_season.races if r.race_no == 7)
    runner = next(run for run in race7.runners if run.brand_no == "J082")

    assert runner.horse_name == "YOUTHFUL SPIRITS"
    assert runner.finish_pos == 6
    assert runner.dead_heat is True
    assert runner.comment is None  # this one also said "No report."


def test_dnf_has_no_position_but_keeps_its_comment(prior_season) -> None:  # type: ignore[no-untyped-def]
    """A horse that did not finish has no placing — but its comment is the important one."""
    race5 = next(r for r in prior_season.races if r.race_no == 5)
    runner = next(run for run in race5.runners if run.brand_no == "G144")

    assert runner.horse_name == "INTREPID WINNER"
    assert runner.finish_pos is None
    assert runner.finished is False
    assert runner.comment is not None
    assert "went amiss" in runner.comment


def test_every_runner_has_the_keys_needed_to_join(current_season) -> None:  # type: ignore[no-untyped-def]
    """Without race_no and brand_no a comment cannot be attached to anything."""
    for race in current_season.races:
        for runner in race.runners:
            assert race.race_no >= 1
            assert runner.brand_no
            assert runner.jockey


# ── Markup stability across both seasons ────────────────────────────────────────

# Four real meetings: two per season, including 2024-11-13, which ran only six races
# and appears to have been abandoned partway — a shortened card must parse like any
# other rather than tripping an assumption about how many races a meeting has.
ALL_MEETINGS = [
    ("report_20260426_valid.html", dt.date(2026, 4, 26), 11, 142),
    ("report_20250907_season_opener.html", dt.date(2025, 9, 7), 10, 133),
    ("report_20250312_prior_season.html", dt.date(2025, 3, 12), 9, 106),
    ("report_20241113_prior_season.html", dt.date(2024, 11, 13), 6, 69),
]


@pytest.mark.parametrize(("name", "day", "races", "runners"), ALL_MEETINGS)
def test_every_fixture_parses_with_the_same_code(
    name: str, day: dt.date, races: int, runners: int
) -> None:
    """No per-season branching: one parser handles both seasons' markup."""
    report = parse_meeting_report(load(name), day)

    assert len(report.races) == races
    assert sum(len(r.runners) for r in report.races) == runners
    assert [r.race_no for r in report.races] == list(range(1, races + 1))


@pytest.mark.parametrize(("name", "day", "_races", "_runners"), ALL_MEETINGS)
def test_no_fixture_leaks_the_no_report_sentinel(
    name: str, day: dt.date, _races: int, _runners: int
) -> None:
    report = parse_meeting_report(load(name), day)

    for race in report.races:
        for runner in race.runners:
            assert runner.comment is None or "No report" not in runner.comment


# ── Horse identity ──────────────────────────────────────────────────────────────


def test_horse_id_is_read_from_the_row_link(current_season) -> None:  # type: ignore[no-untyped-def]
    """The visible text gives a brand number; only the link gives the import year."""
    runner = current_season.races[0].runners[0]

    assert runner.horse_id == "HK_2025_L133"
    assert runner.horse_id.endswith(runner.brand_no)


def test_every_runner_carries_a_horse_id(current_season) -> None:  # type: ignore[no-untyped-def]
    for race in current_season.races:
        for runner in race.runners:
            assert runner.horse_id is not None
            assert runner.horse_id.endswith(runner.brand_no), "id and text must agree"
