"""Writing chunks, and not writing them twice.

Embedding is the expensive step in this project — 12,000 comments on 2 CPU cores —
so the store is built around not repeating it. Four rules do all the work:

**The chunk is addressed by its source and its position in it.** ``(source_type,
source_id, chunk_index)`` is unique, so a comment's third sentence can only ever have
one chunk. Idempotency is a database constraint, not a convention some future caller
can forget.

**Text decides whether we re-encode.** A piece whose text is unchanged keeps its
vector untouched, even if the meeting around it was re-ingested. Only an amended
report — HKJC does amend them — pays for a new encoding, and it replaces the old
chunk in place rather than adding a second one, so the corpus never holds both a
claim and its correction. Because the pieces are positional, an amendment that adds a
sentence in the middle re-encodes that sentence and everything after it; that is the
price of an ordinal address, and it is bounded by the length of one comment.

**A comment that loses a sentence loses its chunk.** An amendment can shorten a
comment, and the pieces beyond the new end are deleted in the same transaction.
Without that, a retracted sentence would stay in the corpus, retrievable and citable,
with nothing pointing at it.

**Metadata is refreshed regardless.** A going downgrade published after the meeting
changes what the chunk filters as, not what it means. Updating the JSONB is nearly
free; re-encoding 12,000 unchanged comments to record it would not be.

The upsert goes through ``ON CONFLICT DO UPDATE`` rather than a read-then-write in
Python: T12's backfill is resumable and may be restarted while a previous run is
still finishing, and two writers reaching the same comment must produce one row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import CursorResult, and_, delete, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from paddock.db.models import Chunk, IncidentComment, Meeting, Race
from paddock.embed.chunker import SOURCE_TYPE, ChunkSpec, build_chunks
from paddock.embed.embedder import Embedder


@dataclass(frozen=True)
class EmbedResult:
    """What one call did, in enough detail to be worth logging.

    The counts are chunks, not comments — one comment is several chunks since
    Checkpoint A (ADR-003), and ``comments`` is carried separately so an integrity
    check can still compare the two. ``embedded + unchanged == total``. A backfill
    that reports ``embedded=0`` across a whole season is not broken — it is the
    resumability working.
    """

    comments: int
    """Comments considered."""
    total: int
    """Chunks those comments describe."""
    embedded: int
    """Newly encoded — either the chunk did not exist or its text changed."""
    unchanged: int
    """Text identical to what is stored; the vector was reused, metadata refreshed."""
    deleted: int
    """Stored chunks whose sentence no longer exists — an amendment shortened it."""


def embed_meeting(session: Session, *, meeting_id: int, embedder: Embedder) -> EmbedResult:
    """Embed every incident comment of one meeting, skipping unchanged text.

    Args:
        session: an open transaction; the caller commits.
        meeting_id: the meeting to embed.
        embedder: any `Embedder` — the real model in production, a fake in tests.

    Raises:
        LookupError: no such meeting.
        ValueError: a comment has empty text (see `chunker.build_chunks`).
    """
    meeting = session.get(Meeting, meeting_id)
    if meeting is None:
        raise LookupError(f"no meeting with id {meeting_id}")

    comment_count, specs = _specs_for(session, meeting)
    if not specs:
        return EmbedResult(comments=comment_count, total=0, embedded=0, unchanged=0, deleted=0)

    stored = _stored_chunks(session, {spec.source_id for spec in specs})

    stale = [spec for spec in specs if _needs_encoding(spec, stored.get(_key(spec)))]
    fresh = [spec for spec in specs if _key(spec) not in {_key(s) for s in stale}]

    _write(session, stale, embedder.embed([spec.text for spec in stale]))
    _refresh_metadata(fresh, stored)
    deleted = _drop_surplus(session, specs)
    session.flush()

    return EmbedResult(
        comments=comment_count,
        total=len(specs),
        embedded=len(stale),
        unchanged=len(fresh),
        deleted=deleted,
    )


def _specs_for(session: Session, meeting: Meeting) -> tuple[int, list[ChunkSpec]]:
    """Every chunk of every commented runner in the meeting, oldest race first."""
    rows = session.execute(
        select(IncidentComment, Race)
        .join(Race, Race.id == IncidentComment.race_id)
        .where(Race.meeting_id == meeting.id)
        .order_by(Race.race_no, IncidentComment.id)
    ).all()
    specs = [
        spec for comment, race in rows for spec in build_chunks(comment, race=race, meeting=meeting)
    ]
    return len(rows), specs


def _stored_chunks(session: Session, comment_ids: set[int]) -> dict[tuple[int, int], Chunk]:
    """What is already in the corpus for these comments, by (comment, position)."""
    return {
        (chunk.source_id, chunk.chunk_index): chunk
        for chunk in session.scalars(
            select(Chunk).where(
                Chunk.source_type == SOURCE_TYPE,
                Chunk.source_id.in_(comment_ids),
            )
        )
    }


def _key(spec: ChunkSpec) -> tuple[int, int]:
    return (spec.source_id, spec.chunk_index)


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
            chunk_index=spec.chunk_index,
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


def _refresh_metadata(specs: list[ChunkSpec], stored: dict[tuple[int, int], Chunk]) -> None:
    """Bring filter metadata up to date without touching the vector."""
    for spec in specs:
        chunk = stored[_key(spec)]
        if chunk.chunk_meta != spec.chunk_meta:
            chunk.chunk_meta = spec.chunk_meta


def _drop_surplus(session: Session, specs: list[ChunkSpec]) -> int:
    """Delete chunks past the end of a comment that got shorter.

    One statement rather than a delete per comment: the amendment case is rare, and
    the common case must not cost a round trip per commented runner.
    """
    highest: dict[int, int] = {}
    for spec in specs:
        highest[spec.source_id] = max(highest.get(spec.source_id, 0), spec.chunk_index)

    # `session.execute` is typed as returning a generic Result; a DELETE always
    # returns the cursor form, and the row count is the only thing wanted from it.
    result = cast(
        "CursorResult[Any]",
        session.execute(
            delete(Chunk).where(
                Chunk.source_type == SOURCE_TYPE,
                or_(
                    *(
                        and_(Chunk.source_id == source_id, Chunk.chunk_index > last)
                        for source_id, last in highest.items()
                    )
                ),
            )
        ),
    )
    return result.rowcount or 0
