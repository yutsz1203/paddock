"""Embedding the whole corpus, and auditing what came out.

`embed_meeting` handles one meeting and raises on anything it does not like. The
corpus is 176 meetings and half an hour of CPU, and it needs the same disposition
`backfill` needed for a season: commit each meeting, write down what happened, and be
resumable when the laptop lid closes at meeting 60.

So what is tested here is not encoding — `test_embed.py` covers that — but what
survives an interruption, what a second run costs, and whether the audit can tell a
finished corpus from one with a hole in it.

Everything runs against a fake embedder and a date far outside HKJC's data. Two of
these tests deliberately break the encoder half-way, which is the only cheap way to
reach the state a real interruption produces.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence

import pytest
from sqlalchemy import func, select
from tests.doubles import FakeEmbedder

from paddock.db.models import Chunk, Horse, IncidentComment, Meeting, Race
from paddock.db.session import session_scope
from paddock.embed.corpus import Coverage, benchmark_search, chunk_coverage, embed_corpus

pytestmark = pytest.mark.integration

# Three consecutive dates far outside HKJC's real data, so a failed run cannot touch
# an ingested meeting. Consecutive so `since`/`until` can select a subset of them.
DATES = [dt.date(2099, 1, 3), dt.date(2099, 1, 10), dt.date(2099, 1, 17)]
WINDOW = {"since": DATES[0], "until": DATES[-1]}

# Comment ids nothing will ever hold, so an orphan seeded here cannot collide with
# a real one that another test module is using.
ORPHAN_IDS = [999_000_001, 999_000_002]

# One comment per meeting, each naming its meeting so a failing encoder can be aimed
# at exactly one of them.
COMMENTS = {
    DATES[0]: "Meeting one runner was hampered approaching the 800 Metres.",
    DATES[1]: "Meeting two runner was slow to begin and lost several lengths.",
    DATES[2]: "Meeting three runner raced wide throughout without cover.",
}


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_test_data()
    yield
    _delete_test_data()


def _seed(dates: Sequence[dt.date] = tuple(DATES)) -> None:
    """One meeting per date, each with a single race and a single commented runner."""
    with session_scope() as session:
        for index, race_date in enumerate(dates):
            meeting = Meeting(race_date=race_date, racecourse="ST", going="GOOD")
            session.add(meeting)
            session.flush()

            race = Race(
                meeting_id=meeting.id,
                race_no=1,
                race_class="Class 4",
                distance_m=1200,
                status="finished",
            )
            session.add(race)
            session.flush()

            horse_id = f"HK_2099_Z{index:03d}"
            session.add(Horse(horse_id=horse_id, brand_no=f"Z{index:03d}", name_en=horse_id))
            session.flush()
            session.add(
                IncidentComment(
                    race_id=race.id,
                    horse_id=horse_id,
                    finish_pos=1,
                    text_en=COMMENTS[race_date],
                )
            )


def _delete_test_data() -> None:
    with session_scope() as session:
        meeting_ids = session.scalars(select(Meeting.id).where(Meeting.race_date.in_(DATES))).all()
        if meeting_ids:
            race_ids = session.scalars(
                select(Race.id).where(Race.meeting_id.in_(meeting_ids))
            ).all()
            comment_ids = session.scalars(
                select(IncidentComment.id).where(IncidentComment.race_id.in_(race_ids))
            ).all()
            if comment_ids:
                session.query(Chunk).filter(Chunk.source_id.in_(comment_ids)).delete(
                    synchronize_session=False
                )
            session.query(Meeting).filter(Meeting.id.in_(meeting_ids)).delete(
                synchronize_session=False
            )
        session.query(Horse).filter(Horse.horse_id.like("HK_2099_%")).delete(
            synchronize_session=False
        )
        session.query(Chunk).filter(Chunk.source_id.in_(ORPHAN_IDS)).delete(
            synchronize_session=False
        )


def _coverage() -> Coverage:
    """Coverage over the whole test database, so assertions must be about deltas.

    Other integration modules leave their own comments behind between tests, and an
    absolute count here would pass or fail depending on what ran first.
    """
    with session_scope() as session:
        return chunk_coverage(session)


def _stored_chunks() -> int:
    with session_scope() as session:
        return (
            session.scalar(
                select(func.count(Chunk.id))
                .join(IncidentComment, IncidentComment.id == Chunk.source_id)
                .join(Race, Race.id == IncidentComment.race_id)
                .join(Meeting, Meeting.id == Race.meeting_id)
                .where(Meeting.race_date.in_(DATES))
            )
            or 0
        )


# ── Walking the corpus ──────────────────────────────────────────────────────────


def test_every_meeting_in_the_window_is_embedded() -> None:
    _seed()

    report = embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    assert report.meetings == 3
    assert report.embedded == 3
    assert report.failed == []
    assert _stored_chunks() == 3


def test_the_window_bounds_which_meetings_are_walked() -> None:
    _seed()

    report = embed_corpus(embedder=FakeEmbedder(), since=DATES[1], until=DATES[1])

    assert report.meetings == 1
    assert [outcome.race_date for outcome in report.outcomes] == [DATES[1]]
    assert _stored_chunks() == 1


def test_meetings_are_walked_oldest_first() -> None:
    _seed()

    report = embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    assert [outcome.race_date for outcome in report.outcomes] == DATES


def test_a_second_run_encodes_nothing() -> None:
    _seed()
    embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    second = FakeEmbedder()
    report = embed_corpus(embedder=second, **WINDOW)

    assert second.embedded_texts == []
    assert report.embedded == 0
    assert report.unchanged == 3


def test_progress_is_reported_as_each_meeting_lands() -> None:
    _seed()
    seen: list[dt.date] = []

    embed_corpus(embedder=FakeEmbedder(), on_outcome=lambda o: seen.append(o.race_date), **WINDOW)

    assert seen == DATES


# ── Surviving an interruption ───────────────────────────────────────────────────


def test_a_meeting_that_fails_does_not_stop_the_run() -> None:
    _seed()

    report = embed_corpus(embedder=FakeEmbedder(fail_on="Meeting two"), **WINDOW)

    assert report.embedded == 2
    assert [outcome.race_date for outcome in report.failed] == [DATES[1]]
    assert "encoder refused" in (report.failed[0].error or "")


def test_a_failed_meeting_leaves_no_chunks_behind() -> None:
    _seed()

    embed_corpus(embedder=FakeEmbedder(fail_on="Meeting two"), **WINDOW)

    # The two that worked are committed; the one that died wrote nothing. Without a
    # transaction per meeting, either all three roll back or a half-embedded meeting
    # is indistinguishable from a finished one on the next run.
    assert _stored_chunks() == 2


def test_a_rerun_finishes_the_meeting_that_failed() -> None:
    _seed()
    embed_corpus(embedder=FakeEmbedder(fail_on="Meeting two"), **WINDOW)

    second = FakeEmbedder()
    report = embed_corpus(embedder=second, **WINDOW)

    assert report.embedded == 1
    assert second.embedded_texts == [COMMENTS[DATES[1]]]
    assert _stored_chunks() == 3


# ── Chunks whose comment is gone ────────────────────────────────────────────────


def _orphan(source_id: int) -> None:
    """A chunk citing a comment id that does not exist.

    Not hypothetical: re-ingesting a meeting deletes its comments and writes new ones
    with new ids, and `chunks.source_id` has no foreign key to follow them (T10). The
    corpus carried 217 of these out of the T11 backfill.
    """
    with session_scope() as session:
        session.add(
            Chunk(
                source_type="incident_comment",
                source_id=source_id,
                chunk_index=0,
                text="Was hampered by a comment that no longer exists.",
                chunk_meta={"race_date": DATES[0].isoformat(), "horse_id": "HK_2099_Z000"},
                embedding=[0.0] * 1024,
            )
        )


def _orphan_ids() -> list[int]:
    with session_scope() as session:
        return list(session.scalars(select(Chunk.source_id).where(Chunk.source_id.in_(ORPHAN_IDS))))


def test_a_chunk_whose_comment_is_gone_is_counted_as_an_orphan() -> None:
    _orphan(ORPHAN_IDS[0])

    assert _coverage().orphans >= 1


def test_the_run_deletes_chunks_whose_comment_is_gone() -> None:
    """Nothing else can. `embed_meeting` walks meetings to comments to chunks, so a
    chunk no comment points at is never visited, and it stays retrievable forever."""
    _seed()
    _orphan(ORPHAN_IDS[0])

    report = embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    assert report.orphans_deleted >= 1
    assert _orphan_ids() == []


def test_a_live_comment_keeps_its_chunks() -> None:
    """The prune is keyed on the comment being absent, not on the chunk being old."""
    _seed()
    embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    report = embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    assert report.orphans_deleted == 0
    assert _stored_chunks() == 3


# ── Auditing the result ─────────────────────────────────────────────────────────


def test_coverage_counts_the_comments_that_have_no_chunk() -> None:
    before = _coverage()
    _seed()

    seeded = _coverage()
    assert seeded.comments == before.comments + 3
    assert seeded.without_chunks == before.without_chunks + 3
    assert not seeded.complete

    embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    embedded = _coverage()
    assert embedded.without_chunks == before.without_chunks
    assert embedded.chunks == before.chunks + 3


def test_the_benchmark_times_one_search_per_query_per_repeat() -> None:
    _seed()
    embed_corpus(embedder=FakeEmbedder(), **WINDOW)

    with session_scope() as session:
        latency = benchmark_search(
            session,
            queries=["trouble in running", "slow to begin"],
            embedder=FakeEmbedder(),
            repeats=3,
        )

    assert latency.samples == 6
    assert 0 < latency.p50_ms <= latency.p95_ms


def test_the_benchmark_embeds_each_query_once_before_timing() -> None:
    _seed()
    embed_corpus(embedder=FakeEmbedder(), **WINDOW)
    embedder = FakeEmbedder()

    with session_scope() as session:
        benchmark_search(session, queries=["trouble in running"], embedder=embedder, repeats=5)

    # Five searches, one encode. Timing the model alongside the index would measure
    # bge-m3 on CPU, which is not what the p95 budget is about.
    assert embedder.embedded_texts == ["trouble in running"]


def test_the_benchmark_refuses_an_empty_query_list() -> None:
    with session_scope() as session, pytest.raises(ValueError, match="at least one query"):
        benchmark_search(session, queries=[], embedder=FakeEmbedder(), repeats=1)
