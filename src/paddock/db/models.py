"""Database schema.

Two design choices worth knowing before reading:

**Natural keys where the source has one.** ``horses.horse_id`` is HKJC's own
``HK_2021_G095``, used directly as the primary key rather than a surrogate integer.
The brand number inside it (``G095``) appears in the incident-report text itself, so
the corpus joins to this table without a lookup step.

**Uniqueness constraints are the idempotency mechanism.** Ingesting a meeting twice
must change nothing, and that guarantee lives here as ``UniqueConstraint`` plus
``ON CONFLICT DO UPDATE`` in the pipeline — not in application logic that could be
bypassed by a future caller.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024  # BAAI/bge-m3

RACE_STATUSES = ("declared", "finished", "abandoned")
INGEST_STATUSES = ("running", "ok", "failed", "fallback_detected", "report_pending")


class Base(DeclarativeBase):
    # Maps Python annotations to column types, so models can write
    # `Mapped[dict[str, Any]]` and get JSONB without repeating the type each time.
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSONB,
        list[float]: JSON,
        list[int]: JSON,
    }


# ── Meetings and races ──────────────────────────────────────────────────────────


class Meeting(Base):
    __tablename__ = "meetings"
    __table_args__ = (UniqueConstraint("race_date", "racecourse", name="uq_meeting_date_course"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    race_date: Mapped[dt.date] = mapped_column(Date, index=True)
    racecourse: Mapped[str] = mapped_column(String(2))  # ST | HV
    going: Mapped[str | None] = mapped_column(String(64))
    source_url: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    races: Mapped[list[Race]] = relationship(back_populates="meeting", cascade="all, delete-orphan")


class Race(Base):
    __tablename__ = "races"
    __table_args__ = (
        UniqueConstraint("meeting_id", "race_no", name="uq_race_meeting_no"),
        CheckConstraint("status IN ('declared', 'finished', 'abandoned')", name="ck_race_status"),
        Index("ix_races_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"))
    race_no: Mapped[int]
    name: Mapped[str | None] = mapped_column(Text)
    race_class: Mapped[str | None] = mapped_column(String(32))
    distance_m: Mapped[int | None]
    track: Mapped[str | None] = mapped_column(String(32))  # Turf | All Weather
    course: Mapped[str | None] = mapped_column(String(16))  # A, B, C, C+3 ...
    going: Mapped[str | None] = mapped_column(String(64))
    prize: Mapped[int | None]
    off_time: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    # 'declared' once the race card is published, 'finished' once results arrive.
    # The lifecycle exists so questions can be asked about races that have not run.
    status: Mapped[str] = mapped_column(String(16), default="declared")

    meeting: Mapped[Meeting] = relationship(back_populates="races")
    runners: Mapped[list[Runner]] = relationship(
        back_populates="race", cascade="all, delete-orphan"
    )


# ── Participants ────────────────────────────────────────────────────────────────


class Horse(Base):
    __tablename__ = "horses"

    # HKJC's own identifier, e.g. 'HK_2021_G095'. Natural key by design.
    horse_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    brand_no: Mapped[str] = mapped_column(String(8), index=True)  # 'G095'
    name_en: Mapped[str | None] = mapped_column(String(128), index=True)
    name_zh: Mapped[str | None] = mapped_column(String(128), index=True)
    country_origin: Mapped[str | None] = mapped_column(String(32))
    colour: Mapped[str | None] = mapped_column(String(64))
    sex: Mapped[str | None] = mapped_column(String(32))
    sire: Mapped[str | None] = mapped_column(String(128))
    dam: Mapped[str | None] = mapped_column(String(128))
    retired: Mapped[bool] = mapped_column(default=False)


class HorseAlias(Base):
    """Prior names. HK horses are renamed mid-career, so name lookups must survive it."""

    __tablename__ = "horse_aliases"
    __table_args__ = (UniqueConstraint("horse_id", "name", "lang", name="uq_alias"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    horse_id: Mapped[str] = mapped_column(ForeignKey("horses.horse_id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128), index=True)
    lang: Mapped[str] = mapped_column(String(4))  # en | zh
    valid_from: Mapped[dt.date | None] = mapped_column(Date)
    valid_to: Mapped[dt.date | None] = mapped_column(Date)


class Jockey(Base):
    __tablename__ = "jockeys"
    __table_args__ = (UniqueConstraint("name_en", name="uq_jockey_name_en"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name_en: Mapped[str] = mapped_column(String(128), index=True)
    name_zh: Mapped[str | None] = mapped_column(String(128), index=True)


class Trainer(Base):
    __tablename__ = "trainers"
    __table_args__ = (UniqueConstraint("name_en", name="uq_trainer_name_en"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name_en: Mapped[str] = mapped_column(String(128), index=True)
    name_zh: Mapped[str | None] = mapped_column(String(128), index=True)


# ── Runners ─────────────────────────────────────────────────────────────────────


class Runner(Base):
    """One horse in one race.

    Created when the race card is published, with every result column NULL, and
    filled in once the race is run. Nullability is therefore load-bearing, not
    incidental: a NULL ``finish_pos`` means "has not run yet", never "unknown".
    """

    __tablename__ = "runners"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", name="uq_runner_race_horse"),
        Index("ix_runners_horse", "horse_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    horse_id: Mapped[str] = mapped_column(ForeignKey("horses.horse_id", ondelete="CASCADE"))

    # Known at declaration time
    horse_no: Mapped[int | None]
    draw: Mapped[int | None]
    jockey_id: Mapped[int | None] = mapped_column(ForeignKey("jockeys.id"))
    trainer_id: Mapped[int | None] = mapped_column(ForeignKey("trainers.id"))

    # Two different quantities that were previously named 'weight' and
    # 'actual_weight' — an invitation to confuse them. One is what the horse
    # carries (~122 lb), the other is the horse's own body weight (~1133 lb).
    carried_weight_lb: Mapped[int | None]
    declared_horse_weight_lb: Mapped[int | None]

    # Carried weight is already net of an apprentice's allowance, so this is the
    # only route back to the weight the handicapper allotted:
    #     allotted = carried_weight_lb + jockey_claim
    # It is also a feature in its own right for a future model.
    jockey_claim: Mapped[int] = mapped_column(default=0)

    rating: Mapped[int | None]
    scratched: Mapped[bool] = mapped_column(default=False)

    # NULL until the race is run
    finish_pos: Mapped[int | None]
    finish_time_s: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float | None] = mapped_column(Float)
    win_odds: Mapped[float | None] = mapped_column(Float)

    # Section count varies with distance (3 at 1000m, 6 at 2400m), so these are
    # arrays rather than the fixed sec_1..sec_n columns of the original spec —
    # fixed columns would be mostly NULL and would break on an unusual distance.
    sectional_times: Mapped[list[float] | None]
    sectional_positions: Mapped[list[int] | None]

    race: Mapped[Race] = relationship(back_populates="runners")


# ── Corpus ──────────────────────────────────────────────────────────────────────


class IncidentComment(Base):
    """A stewards' narrative comment about one runner in one race.

    Absence is meaningful: roughly half of runners receive no comment, and that
    means a clean run. There is no row for those, and the agent must report the
    absence rather than inventing one.
    """

    __tablename__ = "incident_comments"
    __table_args__ = (UniqueConstraint("race_id", "horse_id", name="uq_comment_race_horse"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    horse_id: Mapped[str] = mapped_column(ForeignKey("horses.horse_id", ondelete="CASCADE"))
    jockey_id: Mapped[int | None] = mapped_column(ForeignKey("jockeys.id"))
    finish_pos: Mapped[int | None]
    text_en: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class Chunk(Base):
    """An embedded passage plus the metadata used to pre-filter vector search.

    ``chunk_meta`` duplicates fields that also live in relational columns. That is
    deliberate: it lets the metadata filter and the ANN scan run in one statement,
    which is what makes hybrid retrieval a single query instead of two result sets
    reconciled in Python.

    A chunk is addressed by ``(source_type, source_id, chunk_index)``: one comment
    yields one chunk per sentence-sized piece (Checkpoint A, ADR-003), and the
    position within the comment is what separates them. ``source_id`` alone is still
    what a citation resolves — the reader is shown the sentence and can check the
    whole comment behind it.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("source_type", "source_id", "chunk_index", name="uq_chunk_source"),
        Index("ix_chunks_meta", "chunk_meta", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32))  # 'incident_comment'
    source_id: Mapped[int]
    chunk_index: Mapped[int] = mapped_column(default=0)  # 0-based, in reading order
    text: Mapped[str] = mapped_column(Text)
    chunk_meta: Mapped[dict[str, Any]] = mapped_column(JSONB)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))


# ── Ingestion bookkeeping ───────────────────────────────────────────────────────


class IngestRun(Base):
    __tablename__ = "ingest_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'ok', 'failed', 'fallback_detected', 'report_pending')",
            name="ck_ingest_status",
        ),
        Index("ix_ingest_runs_date", "race_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(64))
    race_date: Mapped[dt.date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Watermark(Base):
    """Last successfully ingested date per source — drives `ingest since`."""

    __tablename__ = "watermarks"

    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_race_date: Mapped[dt.date | None] = mapped_column(Date)
    last_run_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
