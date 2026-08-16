"""The minimal retrieval pair: one SQL query, one vector search.

These two functions are the whole point of the project's central claim — that some
questions are SQL questions and some are retrieval questions, and an agent that knows
which is which beats one that embeds everything. So the tests are about the seam:

**`recent_runs` is form, not prose.** "Last 5 runs, newest first" is `ORDER BY date
DESC LIMIT 5`. No embedding can express it, and no amount of retrieval tuning will
make it reliable.

**`search_comments` filters before the ANN scan, not after.** The sharpest test here
seeds a horse whose comments are all semantically *distant* from the query, surrounded
by other horses' comments that are close. A post-filter implementation returns nothing;
a pre-filter returns the horse's three nearest. That difference is invisible on a
seeded meeting of ten rows and fatal on 12,000.

A stub embedder with hand-placed vectors does the semantic work, because the
assertions need distances that are known rather than plausible. bge-m3's real
behaviour is tested in `test_embed.py` under the `model` marker.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator, Sequence

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from paddock.db.models import (
    EMBEDDING_DIM,
    Chunk,
    Horse,
    IncidentComment,
    Jockey,
    Meeting,
    Race,
    Runner,
    Trainer,
)
from paddock.db.session import session_scope
from paddock.embed.store import embed_meeting
from paddock.retrieval.sql_tool import recent_runs
from paddock.retrieval.vector_tool import search_comments

pytestmark = pytest.mark.integration

# Well outside HKJC's real data, so a failed run cannot corrupt an ingested meeting.
SEASON_START = dt.date(2098, 9, 6)
TARGET = "HK_2099_Z001"
OTHER = "HK_2099_Z002"
QUERY = "was the horse hampered?"


def _axis_vector(similarity: float) -> list[float]:
    """A unit vector whose cosine similarity to `_QUERY_VECTOR` is `similarity`.

    Two dimensions carry all the meaning and the remaining 1022 are zero — enough to
    order results exactly, which is what these assertions need.
    """
    vector = [similarity, math.sqrt(max(0.0, 1.0 - similarity**2))]
    return vector + [0.0] * (EMBEDDING_DIM - 2)


_QUERY_VECTOR = _axis_vector(1.0)


class StubEmbedder:
    """Maps known text to hand-placed vectors; anything unseen sits at the far end."""

    dim = EMBEDDING_DIM

    def __init__(self, placements: dict[str, float]) -> None:
        self.placements = placements

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_axis_vector(self.placements.get(text, 0.0)) for text in texts]


# ── Seeding ─────────────────────────────────────────────────────────────────────


def _meeting(session: Session, race_date: dt.date, racecourse: str = "ST") -> Meeting:
    meeting = Meeting(race_date=race_date, racecourse=racecourse, going="GOOD")
    session.add(meeting)
    session.flush()
    return meeting


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_test_data()
    yield
    _delete_test_data()


def _delete_test_data() -> None:
    with session_scope() as session:
        meeting_ids = list(
            session.scalars(select(Meeting.id).where(Meeting.race_date >= SEASON_START))
        )
        race_ids = list(session.scalars(select(Race.id).where(Race.meeting_id.in_(meeting_ids))))
        comment_ids = list(
            session.scalars(select(IncidentComment.id).where(IncidentComment.race_id.in_(race_ids)))
        )
        if comment_ids:
            session.query(Chunk).filter(Chunk.source_id.in_(comment_ids)).delete(
                synchronize_session=False
            )
        if meeting_ids:
            session.query(Meeting).filter(Meeting.id.in_(meeting_ids)).delete(
                synchronize_session=False
            )
        session.query(Horse).filter(Horse.horse_id.like("HK_2099_%")).delete(
            synchronize_session=False
        )
        session.query(Jockey).filter(Jockey.name_en.like("Z Test%")).delete(
            synchronize_session=False
        )
        session.query(Trainer).filter(Trainer.name_en.like("Z Test%")).delete(
            synchronize_session=False
        )


def _seed_form() -> None:
    """Six finished runs for TARGET, plus the three cases that must not appear:
    an unrun declared race, a scratching, and another horse's run."""
    with session_scope() as session:
        for horse_id in (TARGET, OTHER):
            session.add(Horse(horse_id=horse_id, brand_no=horse_id[-4:], name_en=horse_id))
        jockey = Jockey(name_en="Z Testrider")
        trainer = Trainer(name_en="Z Testhandler")
        session.add_all([jockey, trainer])
        session.flush()

        for index in range(6):
            meeting = _meeting(session, SEASON_START + dt.timedelta(days=7 * index))
            race = Race(
                meeting_id=meeting.id,
                race_no=index + 1,
                race_class="Class 4",
                distance_m=1200 if index % 2 == 0 else 1400,
                course="A",
                going="GOOD",
                status="finished",
            )
            session.add(race)
            session.flush()
            session.add(
                Runner(
                    race_id=race.id,
                    horse_id=TARGET,
                    horse_no=index + 1,
                    draw=index + 1,
                    jockey_id=jockey.id,
                    trainer_id=trainer.id,
                    carried_weight_lb=126,
                    finish_pos=index + 1,
                    finish_time_s=69.5 + index,
                    margin=float(index),
                    win_odds=4.5 + index,
                )
            )
            # A rival, so field size is not trivially 1.
            session.add(Runner(race_id=race.id, horse_id=OTHER, horse_no=9, finish_pos=9))

        # Declared but not yet run — results are NULL, and it is not form.
        upcoming = _meeting(session, SEASON_START + dt.timedelta(days=70))
        declared = Race(meeting_id=upcoming.id, race_no=1, distance_m=1200, status="declared")
        session.add(declared)
        session.flush()
        session.add(Runner(race_id=declared.id, horse_id=TARGET, horse_no=3, draw=3))

        # Declared, then scratched — the horse never left the barrier.
        scratched_meeting = _meeting(session, SEASON_START + dt.timedelta(days=77))
        scratched_race = Race(
            meeting_id=scratched_meeting.id, race_no=1, distance_m=1200, status="finished"
        )
        session.add(scratched_race)
        session.flush()
        session.add(Runner(race_id=scratched_race.id, horse_id=TARGET, horse_no=4, scratched=True))


# ── recent_runs ─────────────────────────────────────────────────────────────────


def test_recent_runs_are_newest_first() -> None:
    _seed_form()

    with session_scope() as session:
        runs = recent_runs(session, horse_id=TARGET, n=6)

    dates = [run.race_date for run in runs]
    assert dates == sorted(dates, reverse=True)


def test_recent_runs_respects_n() -> None:
    _seed_form()

    with session_scope() as session:
        runs = recent_runs(session, horse_id=TARGET, n=3)

    assert len(runs) == 3
    assert runs[0].race_date == SEASON_START + dt.timedelta(days=35)


def test_recent_runs_carries_the_form_line() -> None:
    _seed_form()

    with session_scope() as session:
        run = recent_runs(session, horse_id=TARGET, n=1)[0]

    assert run.racecourse == "ST"
    assert run.distance_m == 1400
    assert run.race_class == "Class 4"
    assert run.going == "GOOD"
    assert run.draw == 6
    assert run.carried_weight_lb == 126
    assert run.jockey == "Z Testrider"
    assert run.trainer == "Z Testhandler"
    assert run.finish_pos == 6
    assert run.win_odds == pytest.approx(9.5)


def test_recent_runs_reports_the_field_size() -> None:
    """ "4th" means nothing without "of 12" — a placing in a five-horse field and one
    in a full field are different facts, and the agent will quote whichever it gets."""
    _seed_form()

    with session_scope() as session:
        run = recent_runs(session, horse_id=TARGET, n=1)[0]

    assert run.field_size == 2


def test_an_unrun_race_is_not_form() -> None:
    """A declared runner has NULL results. Counting it as a run would report a horse
    as unplaced in a race that has not happened."""
    _seed_form()

    with session_scope() as session:
        runs = recent_runs(session, horse_id=TARGET, n=10)

    assert all(run.finish_pos is not None for run in runs)
    assert len(runs) == 6


def test_a_scratched_runner_is_not_form() -> None:
    _seed_form()

    with session_scope() as session:
        runs = recent_runs(session, horse_id=TARGET, n=10)

    assert SEASON_START + dt.timedelta(days=77) not in [run.race_date for run in runs]


def test_a_debutant_has_no_form_rather_than_an_error() -> None:
    """Every race card has one. "No prior runs" is the correct answer, and the agent
    must be able to say it without an exception 200ms earlier."""
    with session_scope() as session:
        assert recent_runs(session, horse_id="HK_2099_Z999", n=5) == []


def test_recent_runs_rejects_a_nonsense_n() -> None:
    with session_scope() as session, pytest.raises(ValueError, match="n"):
        recent_runs(session, horse_id=TARGET, n=0)


def test_a_horse_id_is_a_bound_parameter_not_sql() -> None:
    """The agent chooses which query to run; it never composes one. This asserts the
    boundary holds even if a horse name arrives from a user prompt."""
    _seed_form()
    injection = "HK_2099_Z001'; DROP TABLE runners; --"

    with session_scope() as session:
        assert recent_runs(session, horse_id=injection, n=5) == []

    with session_scope() as session:
        assert session.scalar(select(Runner).limit(1)) is not None


# ── search_comments ─────────────────────────────────────────────────────────────


def _seed_comments() -> StubEmbedder:
    """TARGET's comments are far from the query; five rivals' comments are close.

    This is the arrangement that separates a pre-filter from a post-filter: the
    global top-5 contains none of TARGET's comments.
    """
    # The question sits at the far end of the axis; every comment is placed relative
    # to it, so "nearest" here is a fact rather than a hope about a real model.
    placements: dict[str, float] = {QUERY: 1.0}
    with session_scope() as session:
        session.add(Horse(horse_id=TARGET, brand_no="Z001", name_en=TARGET))
        meeting = _meeting(session, SEASON_START)

        # A horse can only be commented on once per race (uq_comment_race_horse), so
        # TARGET's three comments are three races — which is what its form looks like
        # anyway.
        races = []
        for race_no in (1, 2, 3):
            race = Race(
                meeting_id=meeting.id,
                race_no=race_no,
                race_class="Class 4",
                distance_m=1200,
                course="A",
                going="GOOD",
                status="finished",
            )
            session.add(race)
            session.flush()
            races.append(race)

        for index, race in enumerate(races):
            text = f"TARGET comment {index}: raced awkwardly in the early stages."
            placements[text] = 0.10 - index * 0.01  # far from the query
            session.add(
                IncidentComment(
                    race_id=race.id, horse_id=TARGET, finish_pos=index + 1, text_en=text
                )
            )

        for index in range(5):
            rival = f"HK_2099_R{index:03d}"
            session.add(Horse(horse_id=rival, brand_no=f"R{index:03d}", name_en=rival))
            session.flush()
            text = f"RIVAL {index} comment: was hampered approaching the 800 Metres."
            placements[text] = 0.99 - index * 0.01  # near the query
            session.add(
                IncidentComment(
                    race_id=races[0].id, horse_id=rival, finish_pos=index + 4, text_en=text
                )
            )

    embedder = StubEmbedder(placements)
    with session_scope() as session:
        meeting_id = session.scalar(select(Meeting.id).where(Meeting.race_date == SEASON_START))
        assert meeting_id is not None
        embed_meeting(session, meeting_id=meeting_id, embedder=embedder)
    return embedder


def test_search_returns_the_nearest_comments_first() -> None:
    embedder = _seed_comments()

    with session_scope() as session:
        hits = search_comments(session, query=QUERY, embedder=embedder, limit=3)

    assert [hit.text for hit in hits] == [
        "RIVAL 0 comment: was hampered approaching the 800 Metres.",
        "RIVAL 1 comment: was hampered approaching the 800 Metres.",
        "RIVAL 2 comment: was hampered approaching the 800 Metres.",
    ]


def test_metadata_filtering_happens_before_the_ann_scan() -> None:
    """The test the plan calls for, and the one that catches the common mistake.

    TARGET's three comments are all further from the query than five rivals', so a
    retrieve-then-filter implementation returns zero rows here. Filtering inside the
    statement returns all three.
    """
    embedder = _seed_comments()

    with session_scope() as session:
        hits = search_comments(session, query=QUERY, embedder=embedder, horse_id=TARGET, limit=5)

    assert len(hits) == 3
    assert {hit.horse_id for hit in hits} == {TARGET}


def test_filters_combine() -> None:
    embedder = _seed_comments()

    with session_scope() as session:
        matching = search_comments(
            session, query=QUERY, embedder=embedder, distance_m=1200, limit=10
        )
        wrong_distance = search_comments(
            session, query=QUERY, embedder=embedder, distance_m=2000, limit=10
        )

    assert len(matching) == 8
    assert wrong_distance == []


def test_a_date_window_is_a_range_not_an_equality() -> None:
    """ "This season" and "since his last win" are both windows. Metadata dates are
    ISO strings precisely so this comparison works without casting."""
    embedder = _seed_comments()

    with session_scope() as session:
        inside = search_comments(
            session,
            query=QUERY,
            embedder=embedder,
            since=SEASON_START - dt.timedelta(days=1),
            limit=10,
        )
        after = search_comments(
            session,
            query=QUERY,
            embedder=embedder,
            since=SEASON_START + dt.timedelta(days=1),
            limit=10,
        )

    assert len(inside) == 8
    assert after == []


def test_hits_carry_the_citation_they_will_be_quoted_as() -> None:
    """An answer without a resolvable source is not an answer (spec §1). The hit has
    to carry enough to render a source card without a second round trip."""
    embedder = _seed_comments()

    with session_scope() as session:
        hit = search_comments(session, query=QUERY, embedder=embedder, limit=1)[0]

    assert hit.comment_id > 0
    assert hit.horse_id.startswith("HK_2099_")
    assert hit.race_date == SEASON_START
    assert hit.racecourse == "ST"
    assert hit.race_no == 1
    assert hit.finish_pos is not None
    assert 0.0 <= hit.distance <= 2.0


def test_search_respects_the_limit() -> None:
    embedder = _seed_comments()

    with session_scope() as session:
        hits = search_comments(session, query=QUERY, embedder=embedder, limit=2)

    assert len(hits) == 2


def test_search_rejects_a_nonsense_limit() -> None:
    embedder = StubEmbedder({})

    with session_scope() as session, pytest.raises(ValueError, match="limit"):
        search_comments(session, query=QUERY, embedder=embedder, limit=0)


def test_an_empty_query_is_refused() -> None:
    """An empty string embeds to a vector that is weakly near everything, so it
    returns confident-looking nonsense rather than nothing."""
    embedder = StubEmbedder({})

    with session_scope() as session, pytest.raises(ValueError, match="query"):
        search_comments(session, query="   ", embedder=embedder)


def test_a_filter_value_is_a_bound_parameter_not_sql() -> None:
    _seed_comments()
    embedder = StubEmbedder({})
    injection = "HK_2099_Z001'; DROP TABLE chunks; --"

    with session_scope() as session:
        assert search_comments(session, query=QUERY, embedder=embedder, horse_id=injection) == []

    with session_scope() as session:
        assert session.scalar(select(Chunk).limit(1)) is not None
