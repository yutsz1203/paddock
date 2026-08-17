"""Embedding a meeting into the chunk store.

Most of this runs against a fake embedder: what is being tested is the bookkeeping —
one chunk per sentence-sized piece, no duplicates on a re-run, metadata kept current,
and a shortened comment losing the chunks it no longer has — and a fake makes those
assertions exact and fast. The real model is exercised separately, under
the ``model`` marker, because the questions it answers are different ones: does a
Chinese query actually find its English source, and does a meeting embed inside the
time budget.

Everything here seeds the database directly. The ingestion pipeline that would
normally produce these rows is T10; embedding does not wait for it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import time
from collections.abc import Iterator, Sequence

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from paddock.db.models import Chunk, Horse, IncidentComment, Meeting, Race
from paddock.db.session import session_scope
from paddock.embed.embedder import EMBEDDING_DIM
from paddock.embed.store import embed_meeting

pytestmark = pytest.mark.integration

# Far outside HKJC's real data, so a failed run cannot corrupt an ingested meeting.
TEST_DATE = dt.date(2099, 1, 3)
TEST_HORSES = ["HK_2099_Z001", "HK_2099_Z002", "HK_2099_Z003"]

COMMENTS = [
    "Was hampered approaching the 800 Metres and lost ground.",
    "Was slow to begin and lost several lengths at the start.",
    # Two sentences, two topics — the shape ADR-003 exists for. Everything asserting a
    # chunk count below depends on this being the only multi-sentence comment here.
    "Veterinary examination after the race revealed no abnormality. "
    "The trainer was advised the horse must trial before racing again.",
]

EXPECTED_CHUNKS = [
    COMMENTS[0],
    COMMENTS[1],
    "Veterinary examination after the race revealed no abnormality.",
    "The trainer was advised the horse must trial before racing again.",
]


class FakeEmbedder:
    """Deterministic vectors from a hash — same text, same vector, no model.

    Distances between these are meaningless, which is the point: any test that needs
    real semantics must ask for the real model rather than quietly passing here.
    """

    dim = EMBEDDING_DIM

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(EMBEDDING_DIM)]

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def _seed(comments: Sequence[str] = tuple(COMMENTS)) -> int:
    """Create one meeting, one race, and one commented runner per comment."""
    with session_scope() as session:
        meeting = Meeting(race_date=TEST_DATE, racecourse="ST", going="GOOD")
        session.add(meeting)
        session.flush()

        race = Race(
            meeting_id=meeting.id,
            race_no=1,
            race_class="Class 4",
            distance_m=1200,
            course="A",
            status="finished",
        )
        session.add(race)
        session.flush()

        for index, text in enumerate(comments):
            horse_id = TEST_HORSES[index % len(TEST_HORSES)]
            if session.get(Horse, horse_id) is None:
                session.add(Horse(horse_id=horse_id, brand_no=horse_id[-4:], name_en=horse_id))
                session.flush()
            # One comment per (race, horse) is the unique constraint, so a longer
            # list of comments needs a horse per comment.
            session.add(
                IncidentComment(
                    race_id=race.id,
                    horse_id=horse_id,
                    finish_pos=index + 1,
                    text_en=text,
                )
            )

        return meeting.id


def _seed_many(count: int) -> int:
    """A meeting the size of a real one — ~68 commented runners across 10 races."""
    with session_scope() as session:
        meeting = Meeting(race_date=TEST_DATE, racecourse="ST", going="GOOD")
        session.add(meeting)
        session.flush()

        for index in range(count):
            race = Race(
                meeting_id=meeting.id,
                race_no=index + 1,
                race_class="Class 4",
                distance_m=1200,
                status="finished",
            )
            session.add(race)
            session.flush()

            horse_id = f"HK_2099_Y{index:03d}"
            session.add(Horse(horse_id=horse_id, brand_no=f"Y{index:03d}", name_en=horse_id))
            session.flush()
            session.add(
                IncidentComment(
                    race_id=race.id,
                    horse_id=horse_id,
                    finish_pos=1,
                    text_en=f"Runner {index} was hampered approaching the 800 Metres.",
                )
            )

        return meeting.id


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_test_data()
    yield
    _delete_test_data()


def _delete_test_data() -> None:
    with session_scope() as session:
        meeting_ids = session.scalars(
            select(Meeting.id).where(Meeting.race_date == TEST_DATE)
        ).all()
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
            # Races and comments cascade from the meeting.
            session.query(Meeting).filter(Meeting.id.in_(meeting_ids)).delete(
                synchronize_session=False
            )
        session.query(Horse).filter(Horse.horse_id.like("HK_2099_%")).delete(
            synchronize_session=False
        )


# ── Bookkeeping ─────────────────────────────────────────────────────────────────


def test_one_chunk_per_sentence() -> None:
    """Single-sentence comments give one chunk each; the two-sentence one gives two,
    both addressed to the same comment. That last pair is the whole of ADR-003."""
    meeting_id = _seed()

    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    assert result.comments == len(COMMENTS)
    assert result.total == len(COMMENTS) + 1  # the last comment is two sentences
    assert result.embedded == result.total

    with session_scope() as session:
        stored = _test_chunks(session)
        assert {chunk.text for chunk in stored} == set(EXPECTED_CHUNKS)
        split = [chunk for chunk in stored if chunk.source_id == _comment_id(session, COMMENTS[2])]
        assert sorted(chunk.chunk_index for chunk in split) == [0, 1]


def test_a_shortened_comment_loses_its_surplus_chunks() -> None:
    """HKJC amendments cut text as well as add it. A retracted sentence that stayed
    behind would be retrievable and citable with nothing pointing at it."""
    meeting_id = _seed()

    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    with session_scope() as session:
        comment_id = _comment_id(session, COMMENTS[2])
        comment = session.get(IncidentComment, comment_id)
        assert comment is not None
        comment.text_en = "Veterinary examination after the race revealed no abnormality."

    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    assert result.deleted == 1
    with session_scope() as session:
        chunks = session.scalars(select(Chunk).where(Chunk.source_id == comment_id)).all()
        assert [chunk.chunk_index for chunk in chunks] == [0]


def test_chunks_carry_the_metadata_retrieval_filters_on() -> None:
    meeting_id = _seed(comments=[COMMENTS[0]])

    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    with session_scope() as session:
        chunk = session.scalar(select(Chunk).where(Chunk.text == COMMENTS[0]))
        assert chunk is not None
        assert chunk.chunk_meta["horse_id"] == TEST_HORSES[0]
        assert chunk.chunk_meta["race_date"] == "2099-01-03"
        assert chunk.chunk_meta["racecourse"] == "ST"
        assert chunk.chunk_meta["distance_m"] == 1200
        assert chunk.chunk_meta["going"] == "GOOD"
        assert chunk.chunk_meta["finish_pos"] == 1


def test_re_embedding_changes_nothing() -> None:
    """Same input, same chunk row — the guarantee T12's resumable backfill rests on."""
    meeting_id = _seed()

    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())
    with session_scope() as session:
        before = {
            (c.source_id, c.chunk_index): (c.id, list(c.embedding)) for c in _test_chunks(session)
        }

    second = FakeEmbedder()
    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=second)

    assert result.embedded == 0
    assert result.unchanged == len(EXPECTED_CHUNKS)
    assert second.embedded_texts == []  # unchanged text is never re-encoded

    with session_scope() as session:
        after = {
            (c.source_id, c.chunk_index): (c.id, list(c.embedding)) for c in _test_chunks(session)
        }
    assert after.keys() == before.keys()
    assert [after[k][0] for k in after] == [before[k][0] for k in after]


def test_an_edited_comment_is_re_embedded_in_place() -> None:
    """HKJC amends reports. The chunk must follow the text without becoming a second
    row, or the corpus would hold both the old claim and the correction."""
    meeting_id = _seed()

    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    with session_scope() as session:
        comment = session.scalar(
            select(IncidentComment).where(IncidentComment.text_en == COMMENTS[0])
        )
        assert comment is not None
        comment_id = comment.id
        comment.text_en = "Was hampered on straightening and became unbalanced."

    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    assert result.embedded == 1
    assert result.unchanged == len(EXPECTED_CHUNKS) - 1

    with session_scope() as session:
        chunks = session.scalars(select(Chunk).where(Chunk.source_id == comment_id)).all()
        assert len(chunks) == 1
        assert chunks[0].text == "Was hampered on straightening and became unbalanced."


def test_corrected_metadata_updates_without_re_encoding() -> None:
    """A going downgrade published after the fact changes the filter, not the text.
    Re-encoding 12,000 unchanged comments to record it would cost an hour of CPU."""
    meeting_id = _seed(comments=[COMMENTS[0]])

    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    with session_scope() as session:
        race = session.scalar(select(Race).where(Race.meeting_id == meeting_id))
        assert race is not None
        race.going = "YIELDING"

    second = FakeEmbedder()
    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=second)

    assert second.embedded_texts == []
    with session_scope() as session:
        chunk = session.scalar(select(Chunk).where(Chunk.text == COMMENTS[0]))
        assert chunk is not None
        assert chunk.chunk_meta["going"] == "YIELDING"


def test_a_meeting_with_no_comments_embeds_nothing() -> None:
    """Every runner ran clean. That is a real meeting, not an error."""
    meeting_id = _seed(comments=[])

    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=FakeEmbedder())

    assert result.total == 0
    assert result.embedded == 0
    assert result.comments == 0


def test_an_unknown_meeting_is_refused() -> None:
    with session_scope() as session, pytest.raises(LookupError, match="meeting"):
        embed_meeting(session, meeting_id=-1, embedder=FakeEmbedder())


def _comment_id(session: Session, text: str) -> int:
    comment = session.scalar(select(IncidentComment).where(IncidentComment.text_en == text))
    assert comment is not None
    return comment.id


def _test_chunks(session: Session) -> Sequence[Chunk]:
    comment_ids = session.scalars(
        select(IncidentComment.id)
        .join(Race, Race.id == IncidentComment.race_id)
        .join(Meeting, Meeting.id == Race.meeting_id)
        .where(Meeting.race_date == TEST_DATE)
    ).all()
    return session.scalars(select(Chunk).where(Chunk.source_id.in_(comment_ids))).all()


# ── The real model ──────────────────────────────────────────────────────────────


@pytest.mark.model
def test_a_chinese_query_finds_its_english_source() -> None:
    """The whole reason for bge-m3 over a cheaper English model.

    Mervin reads the Chinese race press; the corpus is English. If a Chinese query
    does not land on the right English comment, the bilingual claim in the spec is
    decoration.
    """
    from paddock.embed.embedder import get_embedder

    meeting_id = _seed()
    embedder = get_embedder()

    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=embedder)

    query = embedder.embed(["馬匹在最後五百米受到阻礙"])[0]  # "hampered in the last 500m"

    with session_scope() as session:
        nearest = session.scalars(
            select(Chunk)
            .where(Chunk.text.in_(COMMENTS))
            .order_by(Chunk.embedding.cosine_distance(query))
        ).all()

    assert nearest[0].text == COMMENTS[0]


@pytest.mark.model
def test_a_full_meeting_embeds_inside_the_budget() -> None:
    """~68 comments is a real meeting. Two minutes on CPU is the deployment budget:
    the Oracle box has 2 ARM OCPUs and the live pipeline embeds a meeting the night
    it runs."""
    from paddock.embed.embedder import get_embedder

    meeting_id = _seed_many(68)
    embedder = get_embedder()  # loading is part of the budget

    started = time.monotonic()
    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=embedder)
    elapsed = time.monotonic() - started

    assert result.embedded == 68
    assert elapsed < 120, f"embedding a meeting took {elapsed:.0f}s"
