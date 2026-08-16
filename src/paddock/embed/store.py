"""Writing chunks, and not writing them twice.

Embedding is the expensive step in this project — 12,000 comments on 2 CPU cores —
so the store is built around not repeating it. Three rules do all the work:

**The chunk is addressed by its source.** ``(source_type, source_id)`` is unique, so
a comment can only ever have one chunk. Idempotency is a database constraint, not a
convention some future caller can forget.

**Text decides whether we re-encode.** A comment whose text is unchanged keeps its
vector untouched, even if the meeting around it was re-ingested. Only an amended
report — HKJC does amend them — pays for a new encoding, and it replaces the old
chunk in place rather than adding a second one, so the corpus never holds both a
claim and its correction.

**Metadata is refreshed regardless.** A going downgrade published after the meeting
changes what the chunk filters as, not what it means. Updating the JSONB is nearly
free; re-encoding 12,000 unchanged comments to record it would not be.

The upsert goes through ``ON CONFLICT DO UPDATE`` rather than a read-then-write in
Python: T12's backfill is resumable and may be restarted while a previous run is
still finishing, and two writers reaching the same comment must produce one row.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from paddock.db.models import Chunk, IncidentComment, Meeting, Race
from paddock.embed.chunker import SOURCE_TYPE, ChunkSpec, build_chunk
from paddock.embed.embedder import Embedder


@dataclass(frozen=True)
class EmbedResult:
    """What one call did, in enough detail to be worth logging.

    ``embedded + unchanged == total``. A backfill that reports ``embedded=0`` across
    a whole season is not broken — it is the resumability working.
    """

    total: int
    """Comments considered."""
    embedded: int
    """Newly encoded — either the chunk did not exist or its text changed."""
    unchanged: int
    """Text identical to what is stored; the vector was reused, metadata refreshed."""


def embed_meeting(session: Session, *, meeting_id: int, embedder: Embedder) -> EmbedResult:
    """Embed every incident comment of one meeting, skipping unchanged text.

    Args:
        session: an open transaction; the caller commits.
        meeting_id: the meeting to embed.
        embedder: any `Embedder` — the real model in production, a fake in tests.

    Raises:
        LookupError: no such meeting.
        ValueError: a comment has empty text (see `chunker.build_chunk`).
    """
    meeting = session.get(Meeting, meeting_id)
    if meeting is None:
        raise LookupError(f"no meeting with id {meeting_id}")

    specs = _specs_for(session, meeting)
    if not specs:
        return EmbedResult(total=0, embedded=0, unchanged=0)

    stored = {
        chunk.source_id: chunk
        for chunk in session.scalars(
            select(Chunk).where(
                Chunk.source_type == SOURCE_TYPE,
                Chunk.source_id.in_([spec.source_id for spec in specs]),
            )
        )
    }

    stale = [spec for spec in specs if _needs_encoding(spec, stored.get(spec.source_id))]
    fresh = [spec for spec in specs if spec.source_id not in {s.source_id for s in stale}]

    _write(session, stale, embedder.embed([spec.text for spec in stale]))
    _refresh_metadata(fresh, stored)
    session.flush()

    return EmbedResult(total=len(specs), embedded=len(stale), unchanged=len(fresh))


def _specs_for(session: Session, meeting: Meeting) -> list[ChunkSpec]:
    """One spec per commented runner in the meeting, oldest race first."""
    rows = session.execute(
        select(IncidentComment, Race)
        .join(Race, Race.id == IncidentComment.race_id)
        .where(Race.meeting_id == meeting.id)
        .order_by(Race.race_no, IncidentComment.id)
    ).all()
    return [build_chunk(comment, race=race, meeting=meeting) for comment, race in rows]


def _needs_encoding(spec: ChunkSpec, stored: Chunk | None) -> bool:
    return stored is None or stored.text != spec.text


def _write(session: Session, specs: list[ChunkSpec], vectors: list[list[float]]) -> None:
    """Insert or replace, one statement per chunk, keyed on the source."""
    if not specs:
        return
    if len(vectors) != len(specs):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(specs)} texts")

    for spec, vector in zip(specs, vectors, strict=True):
        statement = insert(Chunk).values(
            source_type=spec.source_type,
            source_id=spec.source_id,
            text=spec.text,
            chunk_meta=spec.chunk_meta,
            embedding=vector,
        )
        session.execute(
            statement.on_conflict_do_update(
                constraint="uq_chunk_source",
                set_={
                    "text": statement.excluded.text,
                    "chunk_meta": statement.excluded.chunk_meta,
                    "embedding": statement.excluded.embedding,
                },
            )
        )


def _refresh_metadata(specs: list[ChunkSpec], stored: dict[int, Chunk]) -> None:
    """Bring filter metadata up to date without touching the vector."""
    for spec in specs:
        chunk = stored[spec.source_id]
        if chunk.chunk_meta != spec.chunk_meta:
            chunk.chunk_meta = spec.chunk_meta
