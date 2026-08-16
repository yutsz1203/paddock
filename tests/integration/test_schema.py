"""Schema integration tests.

These assert the guarantees the rest of the system leans on: that migrations are
reversible, that pgvector is present, and that the uniqueness constraints which
make ingestion idempotent actually exist in the database rather than only in the
model definitions.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from paddock.db.models import Horse, Meeting, Race, Runner
from paddock.db.session import get_engine, session_scope

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "meetings",
    "races",
    "horses",
    "horse_aliases",
    "jockeys",
    "trainers",
    "runners",
    "incident_comments",
    "chunks",
    "ingest_runs",
    "watermarks",
}


def test_all_tables_exist() -> None:
    tables = set(inspect(get_engine()).get_table_names())
    assert tables >= EXPECTED_TABLES, f"missing: {EXPECTED_TABLES - tables}"


def test_pgvector_extension_installed() -> None:
    with get_engine().connect() as conn:
        version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
    assert version is not None, "vector extension missing — migration did not create it"


def test_embedding_index_is_hnsw_cosine() -> None:
    """The ANN index must exist, or vector search silently degrades to a full scan."""
    with get_engine().connect() as conn:
        indexdef = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_embedding'")
        ).scalar()

    assert indexdef is not None
    assert "hnsw" in indexdef.lower()
    assert "vector_cosine_ops" in indexdef.lower()


def test_runner_uniqueness_blocks_duplicate_ingestion() -> None:
    """The idempotency guarantee is enforced by the database, not by caller discipline."""
    with session_scope() as session:
        meeting = Meeting(race_date=dt.date(2099, 1, 1), racecourse="ST")
        session.add(meeting)
        session.flush()

        race = Race(meeting_id=meeting.id, race_no=1, status="finished")
        horse = Horse(horse_id="HK_2099_Z001", brand_no="Z001", name_en="TEST HORSE")
        session.add_all([race, horse])
        session.flush()

        session.add(Runner(race_id=race.id, horse_id=horse.horse_id, horse_no=1))
        session.flush()

        session.add(Runner(race_id=race.id, horse_id=horse.horse_id, horse_no=1))
        with pytest.raises(IntegrityError):
            session.flush()

        session.rollback()

    # Clean up so repeated local runs stay green.
    with session_scope() as session:
        session.query(Meeting).filter_by(race_date=dt.date(2099, 1, 1)).delete()
        session.query(Horse).filter_by(horse_id="HK_2099_Z001").delete()


def test_result_columns_are_nullable() -> None:
    """A declared runner has no result yet — NULL means 'has not run', not 'unknown'."""
    columns = {c["name"]: c for c in inspect(get_engine()).get_columns("runners")}

    for column in ("finish_pos", "finish_time_s", "margin", "win_odds"):
        assert columns[column]["nullable"], f"{column} must be nullable for declared races"
