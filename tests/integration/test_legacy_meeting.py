"""A meeting from before October 2023, when the report had no runner rows.

The older report gives the card — race numbers, names, classes, distances — and a
paragraph of stewards' prose per race. The runners come from the results pages
instead, so the meeting is stored with results and sectionals and no comments.

Only the report fixture is from that era. The results and sectional fixtures are the
2026-04-26 Race 1 pages, served under the 2023 URLs: the parsers do not read the
date from those pages, and what is under test is where each column comes from.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Iterator

import pytest
from sqlalchemy import select
from tests.doubles import RecordingFetcher

from paddock.db.models import (
    FetchedPage,
    IncidentComment,
    IngestRun,
    Meeting,
    Race,
    Runner,
    Watermark,
)
from paddock.db.session import session_scope
from paddock.ingest import pipeline
from paddock.ingest.pipeline import ingest_meeting
from paddock.ingest.results import parse_race_results
from paddock.ingest.watermark import INCIDENT_REPORT

pytestmark = pytest.mark.integration

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "html"

RACE_DATE = dt.date(2023, 1, 1)
RACECOURSE = "ST"
RACES_IN_CARD = 11


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_everything()
    yield
    _delete_everything()


def _delete_everything() -> None:
    with session_scope() as session:
        session.query(Meeting).filter(Meeting.race_date == RACE_DATE).delete(
            synchronize_session=False
        )
        session.query(IngestRun).filter(IngestRun.race_date == RACE_DATE).delete(
            synchronize_session=False
        )
        session.query(FetchedPage).filter(FetchedPage.url.in_(_urls())).delete(
            synchronize_session=False
        )
        session.query(Watermark).filter(Watermark.source == INCIDENT_REPORT).delete(
            synchronize_session=False
        )


def _urls() -> list[str]:
    url_for = RecordingFetcher().url_for
    urls = [url_for(pipeline.REPORT_PATH, pipeline.report_params(RACE_DATE))]
    for race_no in range(1, RACES_IN_CARD + 1):
        urls.append(
            url_for(pipeline.RESULTS_PATH, pipeline.results_params(RACE_DATE, RACECOURSE, race_no))
        )
        urls.append(
            url_for(pipeline.SECTIONALS_PATH, pipeline.sectionals_params(RACE_DATE, race_no))
        )
    return urls


def _fixture_client() -> RecordingFetcher:
    client = RecordingFetcher()
    client.serve(
        pipeline.REPORT_PATH,
        pipeline.report_params(RACE_DATE),
        (FIXTURES / "report_20230101_legacy.html").read_text(encoding="utf-8"),
    )
    client.serve(
        pipeline.RESULTS_PATH,
        pipeline.results_params(RACE_DATE, RACECOURSE, 1),
        (FIXTURES / "results_20260426_ST_R1.html").read_text(),
    )
    client.serve(
        pipeline.SECTIONALS_PATH,
        pipeline.sectionals_params(RACE_DATE, 1),
        (FIXTURES / "sectional_20260426_R1.html").read_text(),
    )
    empty = (FIXTURES / "results_20260423_no_meeting.html").read_text()
    for race_no in range(2, RACES_IN_CARD + 1):
        client.serve(
            pipeline.RESULTS_PATH, pipeline.results_params(RACE_DATE, RACECOURSE, race_no), empty
        )
        client.serve(
            pipeline.SECTIONALS_PATH, pipeline.sectionals_params(RACE_DATE, race_no), empty
        )
    return client


def _expected_runners() -> int:
    """Race 1's results page, counted the way the pipeline stores it: by horse id."""
    results = parse_race_results((FIXTURES / "results_20260426_ST_R1.html").read_text())
    return sum(1 for runner in results.runners if runner.horse_id)


def test_the_card_comes_from_the_report_and_the_runners_from_the_results() -> None:
    # No racecourse passed: backfill has none, so it must come from the info block.
    result = ingest_meeting(_fixture_client(), RACE_DATE)

    assert result.races == RACES_IN_CARD
    assert result.runners == _expected_runners()
    assert result.comments == 0, "the older report's prose is not split per horse"

    with session_scope() as session:
        meeting = session.scalar(select(Meeting).where(Meeting.race_date == RACE_DATE))
        assert meeting is not None
        assert meeting.racecourse == RACECOURSE
        race = session.scalar(select(Race).where(Race.meeting_id == meeting.id, Race.race_no == 1))
        assert race is not None
        # Name, class and distance from the 2023 report; track from the results page.
        assert (race.name, race.race_class, race.distance_m) == ("YEW HANDICAP", "Class 5", 1400)
        assert race.track == "TURF"
        runners = list(session.scalars(select(Runner).where(Runner.race_id == race.id)))
        assert runners, "race 1 has a results page, so it must have runners"
        assert all(r.finish_time_s is not None for r in runners if r.finish_pos is not None)
        assert any(r.sectional_times for r in runners), "sectionals still join by brand"
        assert (
            session.query(IncidentComment).filter(IncidentComment.race_id == race.id).count() == 0
        )


def test_a_race_without_a_results_page_is_kept_with_no_runners() -> None:
    result = ingest_meeting(_fixture_client(), RACE_DATE)

    assert result.races_without_results == list(range(2, RACES_IN_CARD + 1))


def test_ingesting_an_older_meeting_twice_changes_nothing() -> None:
    ingest_meeting(_fixture_client(), RACE_DATE)
    with session_scope() as session:
        first = sorted(session.scalars(select(Runner.id)))

    ingest_meeting(_fixture_client(), RACE_DATE)
    with session_scope() as session:
        second = sorted(session.scalars(select(Runner.id)))

    assert first == second
