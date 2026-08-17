"""Ingesting a meeting twice, and ingesting one that breaks half-way.

Two guarantees, and they pull in opposite directions.

**The second run must change nothing.** Not "produce the same counts" — the same
rows, with the same primary keys. `chunks.source_id` points at
`incident_comments.id` without a foreign key (the chunk table is addressed by
`(source_type, source_id, chunk_index)`, not by a constraint), so a re-ingest that
reassigned comment ids would silently orphan every embedding that cites one. That is
why these tests compare ids and not just totals.

**A run that fails half-way must change nothing either.** One transaction per
meeting: races 1-4 written and race 5 malformed means no meeting at all, so a retry
starts from a clean slate rather than from four races and a guess.

The fixtures cover one meeting: the incident report carries the full card (11 races,
142 runners, 114 of them commented) and results and sectionals were captured for
Race 1 only, so Race 1 is where enrichment is asserted.
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
from paddock.ingest.date_guard import FallbackDetectedError
from paddock.ingest.pipeline import ingest_meeting
from paddock.ingest.watermark import INCIDENT_REPORT, get_watermark

pytestmark = pytest.mark.integration

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "html"

RACE_DATE = dt.date(2026, 4, 26)
FALLBACK_DATE = dt.date(2026, 4, 23)  # a Thursday — HK never races on one
RACECOURSE = "ST"
RACES_IN_CARD = 11


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_everything()
    yield
    _delete_everything()


def _delete_everything() -> None:
    with session_scope() as session:
        for day in (RACE_DATE, FALLBACK_DATE):
            session.query(Meeting).filter(Meeting.race_date == day).delete(
                synchronize_session=False
            )
            session.query(IngestRun).filter(IngestRun.race_date == day).delete(
                synchronize_session=False
            )
            # The archive outlives a meeting by design, so it has to be cleared here
            # too — otherwise the second test to run makes no requests at all, and
            # every assertion about fetching becomes vacuous.
            session.query(FetchedPage).filter(FetchedPage.url.in_(_urls_for(day))).delete(
                synchronize_session=False
            )
        session.query(Watermark).filter(Watermark.source == INCIDENT_REPORT).delete(
            synchronize_session=False
        )


def _urls_for(day: dt.date) -> list[str]:
    """Every URL a meeting on `day` could be archived under."""
    url_for = RecordingFetcher().url_for
    urls = [url_for(pipeline.REPORT_PATH, pipeline.report_params(day))]
    for race_no in range(1, RACES_IN_CARD + 1):
        urls.append(
            url_for(pipeline.RESULTS_PATH, pipeline.results_params(day, RACECOURSE, race_no))
        )
        urls.append(url_for(pipeline.SECTIONALS_PATH, pipeline.sectionals_params(day, race_no)))
    return urls


def _fixture_client() -> RecordingFetcher:
    """One meeting's pages, as HKJC would serve them.

    Races 2-11 get the "no meeting" results page, which is what that endpoint
    genuinely returns when it has nothing — the fixtures only cover Race 1.
    """
    client = RecordingFetcher()
    client.serve(
        pipeline.REPORT_PATH,
        {"date": "2026/04/26"},
        (FIXTURES / "report_20260426_valid.html").read_text(),
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


def _snapshot() -> dict[str, list[tuple[object, ...]]]:
    """Every row the meeting owns, by primary key, in a comparable shape."""
    with session_scope() as session:
        meeting = session.scalar(select(Meeting).where(Meeting.race_date == RACE_DATE))
        assert meeting is not None
        races = list(
            session.scalars(
                select(Race).where(Race.meeting_id == meeting.id).order_by(Race.race_no)
            )
        )
        race_ids = [race.id for race in races]
        runners = list(
            session.scalars(select(Runner).where(Runner.race_id.in_(race_ids)).order_by(Runner.id))
        )
        comments = list(
            session.scalars(
                select(IncidentComment)
                .where(IncidentComment.race_id.in_(race_ids))
                .order_by(IncidentComment.id)
            )
        )
        return {
            "meeting": [(meeting.id, meeting.race_date, meeting.racecourse, meeting.going)],
            "races": [(r.id, r.race_no, r.name, r.distance_m, r.status) for r in races],
            "runners": [
                (n.id, n.race_id, n.horse_id, n.finish_pos, n.win_odds, n.sectional_times)
                for n in runners
            ],
            "comments": [(c.id, c.race_id, c.horse_id, c.text_en) for c in comments],
        }


# ── The happy path ──────────────────────────────────────────────────────────────


def test_ingesting_a_meeting_writes_the_whole_card() -> None:
    result = ingest_meeting(_fixture_client(), RACE_DATE, RACECOURSE)

    assert result.races == RACES_IN_CARD
    assert result.runners == 142
    assert result.comments == 114, "roughly one runner in five ran clean and gets no row"


def test_results_and_sectionals_enrich_the_race_they_cover() -> None:
    """The report gives the card; the results page gives odds, times and sectionals."""
    ingest_meeting(_fixture_client(), RACE_DATE, RACECOURSE)

    with session_scope() as session:
        winner = session.scalar(
            select(Runner)
            .join(Race, Runner.race_id == Race.id)
            .join(Meeting, Race.meeting_id == Meeting.id)
            .where(Meeting.race_date == RACE_DATE, Race.race_no == 1, Runner.finish_pos == 1)
        )
        assert winner is not None
        assert winner.win_odds is not None
        assert winner.finish_time_s is not None
        assert winner.sectional_times, "sectionals are the only source of per-section times"


def test_a_successful_meeting_advances_the_watermark() -> None:
    ingest_meeting(_fixture_client(), RACE_DATE, RACECOURSE)

    with session_scope() as session:
        assert get_watermark(session, INCIDENT_REPORT) == RACE_DATE


# ── Idempotency ─────────────────────────────────────────────────────────────────


def test_ingesting_twice_leaves_the_database_byte_for_byte_identical() -> None:
    client = _fixture_client()
    ingest_meeting(client, RACE_DATE, RACECOURSE)
    first = _snapshot()

    ingest_meeting(client, RACE_DATE, RACECOURSE)

    assert _snapshot() == first


def test_re_ingesting_keeps_comment_ids_stable() -> None:
    """`chunks.source_id` cites these ids without a foreign key to enforce it."""
    client = _fixture_client()
    ingest_meeting(client, RACE_DATE, RACECOURSE)
    before = {(c[1], c[2]): c[0] for c in _snapshot()["comments"]}

    ingest_meeting(client, RACE_DATE, RACECOURSE)
    after = {(c[1], c[2]): c[0] for c in _snapshot()["comments"]}

    assert after == before, "a reassigned id orphans every embedding that cites it"


def test_the_second_run_makes_no_requests_at_all() -> None:
    """The archive answers everything — a re-run costs HKJC nothing."""
    client = _fixture_client()
    ingest_meeting(client, RACE_DATE, RACECOURSE)
    first_pass = len(client.requests)

    ingest_meeting(client, RACE_DATE, RACECOURSE)

    assert len(client.requests) == first_pass
    assert first_pass == 1 + 2 * RACES_IN_CARD, "one report, plus results and sectionals per race"


# ── Failure part-way ────────────────────────────────────────────────────────────


def test_a_failure_part_way_leaves_no_partial_meeting() -> None:
    client = _fixture_client()
    client.fail(
        pipeline.RESULTS_PATH,
        pipeline.results_params(RACE_DATE, RACECOURSE, 5),
        RuntimeError("connection reset"),
    )

    with pytest.raises(RuntimeError, match="connection reset"):
        ingest_meeting(client, RACE_DATE, RACECOURSE)

    with session_scope() as session:
        assert session.scalar(select(Meeting).where(Meeting.race_date == RACE_DATE)) is None
        assert get_watermark(session, INCIDENT_REPORT) is None, "a failed meeting is not progress"


def test_a_failure_part_way_is_still_recorded_as_a_run() -> None:
    client = _fixture_client()
    client.fail(
        pipeline.RESULTS_PATH,
        pipeline.results_params(RACE_DATE, RACECOURSE, 5),
        RuntimeError("connection reset"),
    )

    with pytest.raises(RuntimeError):
        ingest_meeting(client, RACE_DATE, RACECOURSE)

    with session_scope() as session:
        run = session.scalar(select(IngestRun).where(IngestRun.race_date == RACE_DATE))
        assert run is not None
        assert run.status == "failed"
        assert "connection reset" in (run.error or "")


def test_a_retry_after_a_failure_completes_the_meeting() -> None:
    """The pages fetched before the failure are archived, so the retry re-uses them."""
    client = _fixture_client()
    client.fail(
        pipeline.RESULTS_PATH,
        pipeline.results_params(RACE_DATE, RACECOURSE, 5),
        RuntimeError("connection reset"),
    )
    with pytest.raises(RuntimeError):
        ingest_meeting(client, RACE_DATE, RACECOURSE)

    client.fail_on.clear()
    result = ingest_meeting(client, RACE_DATE, RACECOURSE)

    assert result.races == RACES_IN_CARD
    # Every page is fetched exactly once across both runs, except Race 5's results —
    # the one the first run never got. The four races before it are not re-fetched.
    once = 1 + 2 * RACES_IN_CARD
    assert len(client.requests) == once + 1
    assert len(set(client.requests)) == once


def test_a_served_fallback_writes_nothing() -> None:
    """HKJC answers 200 with the latest meeting for a date that never raced (ADR-002)."""
    client = RecordingFetcher()
    client.serve(
        pipeline.REPORT_PATH,
        pipeline.report_params(FALLBACK_DATE),
        (FIXTURES / "report_20260423_fallback.html").read_text(),
    )

    with pytest.raises(FallbackDetectedError, match="2026-04-23"):
        ingest_meeting(client, FALLBACK_DATE, RACECOURSE)

    with session_scope() as session:
        assert session.scalar(select(Meeting).where(Meeting.race_date == FALLBACK_DATE)) is None
        run = session.scalar(select(IngestRun).where(IngestRun.race_date == FALLBACK_DATE))
        assert run is not None
        assert run.status == "fallback_detected"
        assert client.requests == [
            client.url_for(pipeline.REPORT_PATH, pipeline.report_params(FALLBACK_DATE))
        ], "the guard runs before any results page is fetched"
