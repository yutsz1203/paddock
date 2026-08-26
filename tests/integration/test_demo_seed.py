"""Cutting the corpus down to a slice small enough to commit.

The demo dump is the one artefact in this repository that a stranger runs before
reading anything. It has to be small enough for GitHub, honest about what it holds,
and free of the two things the corpus keeps that must not be republished: HKJC's
archived pages, and every meeting outside the advertised window.

**These tests own the database.** Pruning is defined against every meeting there is,
so a leftover row from another module changes the answer. The suite runs against its
own database (`tests/conftest.py`), which is what makes a full wipe here safe.

The sharpest test is the orphan-chunk one. `chunks.source_id` carries no foreign key
— it is a soft reference, because a chunk can in principle come from a source other
than a comment — so nothing in the database removes a chunk when its comment goes.
A dump built without that step ships vectors that answer questions and then cite a
comment id that does not resolve, which is the exact failure the citation rule exists
to prevent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select

from paddock.db.models import (
    EMBEDDING_DIM,
    Chunk,
    FetchedPage,
    Horse,
    IncidentComment,
    IngestRun,
    Jockey,
    Meeting,
    Race,
    Runner,
    Trainer,
    Watermark,
)
from paddock.db.session import session_scope
from paddock.demo.seed import prune_to_recent_meetings, seed_report

pytestmark = pytest.mark.integration

# Well outside HKJC's real data, so a failed run cannot touch an ingested meeting.
FIRST = dt.date(2098, 9, 6)


def _day(offset: int) -> dt.date:
    """Meeting `offset` of the fake season, one a week."""
    return FIRST + dt.timedelta(days=7 * offset)


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _wipe()
    yield
    _wipe()


def _wipe() -> None:
    with session_scope() as session:
        for model in (Chunk, IncidentComment, Runner, Race, Meeting):
            session.query(model).delete(synchronize_session=False)
        for model in (Horse, Jockey, Trainer, FetchedPage, IngestRun, Watermark):
            session.query(model).delete(synchronize_session=False)


def _seed_meeting(offset: int, *, horse_id: str, jockey: str, trainer: str) -> None:
    """One meeting, one race, one runner, one commented runner, one chunk.

    Every table the prune touches gets exactly one row per meeting, so a count is a
    meeting count and an assertion reads as the number of meetings kept.
    """
    with session_scope() as session:
        horse = session.get(Horse, horse_id) or Horse(horse_id=horse_id, brand_no=horse_id[-4:])
        session.add(horse)
        jockey_row = session.scalar(select(Jockey).where(Jockey.name_en == jockey)) or Jockey(
            name_en=jockey
        )
        trainer_row = session.scalar(select(Trainer).where(Trainer.name_en == trainer)) or Trainer(
            name_en=trainer
        )
        session.add_all([jockey_row, trainer_row])
        session.flush()

        meeting = Meeting(race_date=_day(offset), racecourse="ST")
        race = Race(meeting=meeting, race_no=1, status="finished")
        session.add_all([meeting, race])
        session.flush()

        session.add(
            Runner(
                race_id=race.id,
                horse_id=horse_id,
                jockey_id=jockey_row.id,
                trainer_id=trainer_row.id,
            )
        )
        comment = IncidentComment(
            race_id=race.id, horse_id=horse_id, text_en=f"comment from meeting {offset}"
        )
        session.add(comment)
        session.flush()

        session.add(
            Chunk(
                source_type="incident_comment",
                source_id=comment.id,
                chunk_index=0,
                text=comment.text_en,
                chunk_meta={"race_date": _day(offset).isoformat()},
                embedding=[0.0] * EMBEDDING_DIM,
            )
        )


def _seed_season(meetings: int) -> None:
    for offset in range(meetings):
        _seed_meeting(
            offset,
            horse_id=f"HK_2099_Z{offset:03d}",
            jockey=f"Jockey {offset}",
            trainer=f"Trainer {offset}",
        )


def _count(model: type) -> int:
    with session_scope() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def test_the_slice_keeps_the_most_recent_meetings_and_drops_the_rest() -> None:
    _seed_season(5)

    with session_scope() as session:
        report = prune_to_recent_meetings(session, meetings=2)

    assert report.meetings == 2
    assert report.first_date == _day(3)
    assert report.last_date == _day(4)

    with session_scope() as session:
        kept = sorted(session.scalars(select(Meeting.race_date)))
    assert kept == [_day(3), _day(4)]


def test_a_dropped_meeting_takes_its_races_runners_and_comments_with_it() -> None:
    _seed_season(5)

    with session_scope() as session:
        prune_to_recent_meetings(session, meetings=2)

    assert _count(Race) == 2
    assert _count(Runner) == 2
    assert _count(IncidentComment) == 2


def test_a_chunk_whose_comment_was_dropped_goes_too() -> None:
    """The soft reference. Nothing in the schema removes these, so the prune must.

    A chunk left behind is a vector that still answers a search and then cites a
    comment id that no longer resolves.
    """
    _seed_season(5)

    with session_scope() as session:
        report = prune_to_recent_meetings(session, meetings=2)

    assert report.chunks == 2
    with session_scope() as session:
        orphans = session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(
                Chunk.source_type == "incident_comment",
                ~Chunk.source_id.in_(select(IncidentComment.id)),
            )
        )
    assert orphans == 0


def test_the_slice_carries_no_archived_hkjc_pages() -> None:
    """Spec §436: no bulk HKJC-derived data beyond the demo slice. The archive is
    117 MB of HKJC's own HTML, and republishing it is not what the demo is for."""
    _seed_season(2)
    with session_scope() as session:
        session.add(
            FetchedPage(url="https://hkjc.test.invalid/page", body_gz=b"gz", sha256="0" * 64)
        )

    with session_scope() as session:
        report = prune_to_recent_meetings(session, meetings=2)

    assert report.pages == 0
    assert _count(FetchedPage) == 0


def test_a_horse_with_no_remaining_run_is_dropped_and_one_that_still_runs_is_kept() -> None:
    """So the demo says "I do not know that horse" rather than "that horse has no runs".

    The second sentence is a claim about form. The demo cannot support it for a horse
    whose every start was cut out of the slice.
    """
    _seed_season(3)

    with session_scope() as session:
        prune_to_recent_meetings(session, meetings=1)

    with session_scope() as session:
        kept = sorted(session.scalars(select(Horse.horse_id)))
    assert kept == ["HK_2099_Z002"]


def test_a_jockey_or_trainer_with_no_remaining_runner_is_dropped() -> None:
    _seed_season(3)

    with session_scope() as session:
        report = prune_to_recent_meetings(session, meetings=1)

    assert report.jockeys == 1
    assert report.trainers == 1
    with session_scope() as session:
        assert sorted(session.scalars(select(Jockey.name_en))) == ["Jockey 2"]
        assert sorted(session.scalars(select(Trainer.name_en))) == ["Trainer 2"]


def test_ingest_bookkeeping_is_dropped_so_the_demo_claims_no_history() -> None:
    """A restored snapshot never ran an ingest. A watermark left behind would tell a
    later `ingest since` that dates it does not hold were already done."""
    _seed_season(2)
    with session_scope() as session:
        session.add(
            IngestRun(
                source="incident_report",
                race_date=_day(0),
                status="ok",
                started_at=dt.datetime(2098, 9, 6, tzinfo=dt.UTC),
            )
        )
        session.add(Watermark(source="incident_report", last_race_date=_day(1)))

    with session_scope() as session:
        prune_to_recent_meetings(session, meetings=2)

    assert _count(IngestRun) == 0
    assert _count(Watermark) == 0


def test_asking_for_more_meetings_than_exist_keeps_every_one() -> None:
    _seed_season(3)

    with session_scope() as session:
        report = prune_to_recent_meetings(session, meetings=20)

    assert report.meetings == 3
    assert report.first_date == _day(0)


def test_the_report_counts_what_is_there_without_changing_it() -> None:
    _seed_season(3)

    with session_scope() as session:
        report = seed_report(session)

    assert report.meetings == 3
    assert report.races == 3
    assert report.runners == 3
    assert report.comments == 3
    assert report.chunks == 3
    assert report.horses == 3
    assert report.first_date == _day(0)
    assert report.last_date == _day(2)
    assert _count(Meeting) == 3
