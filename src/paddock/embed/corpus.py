"""Embedding all 176 meetings, and proving afterwards that it worked.

`embed_meeting` handles one meeting and raises on anything it does not like. The
corpus is 17,931 comments and about half an hour of CPU, and it needs the disposition
`ingest.backfill` needed for a season: commit each meeting, write down what happened,
and be resumable when the laptop lid closes at meeting 60.

## Resumability is a property of the store, not of a bookmark

There is no watermark here and no "last meeting embedded" row. `embed_meeting`
already re-encodes only text it has not seen, so a meeting that finished costs two
queries and a sentence split on the second pass — milliseconds — and a meeting that
died half-way costs only the sentences it never reached. A bookmark would add a
second source of truth that can disagree with the corpus, to save something already
close to free.

What resumability does need is a transaction per meeting. `session_scope` commits
each one on the way out, so an interruption leaves finished meetings finished rather
than rolling back an hour of work.

## A failed meeting is recorded, not raised

One meeting that will not encode must not end a run over 176 of them, for the same
reason one unparseable report does not end a backfill. The exception is caught
broadly, which `backfill` deliberately does not do — there the failure surface is
HKJC's markup and it can be enumerated, here it is a 2.2 GB third-party model on CPU
and it cannot. The report is loud instead: failures are named with their dates, and
`paddock embed --all` exits non-zero if there is one.

## Chunks whose comment is gone

Re-ingesting a meeting deletes its comments and writes new ones with new ids, and
`chunks.source_id` carries no foreign key to follow them (T10 kept it that way on
purpose, so a reassignment is visible rather than cascaded away). What that leaves is
a chunk citing a comment id that no longer exists — still indexed, still retrievable,
still carrying metadata that reads as current, and citing a row nobody can resolve.
The T11 backfill left 217 of them.

Nothing else can remove them: `embed_meeting` walks meetings to comments to chunks,
so a chunk no comment points at is never visited. So a corpus run deletes them first,
in a transaction of its own, and says how many. The test for it is
`test_a_live_comment_keeps_its_chunks` — the prune is keyed on the comment being
absent, never on the chunk being old.

## Where the corpus goes next

Embedding happens locally and the box gets a `pg_dump` restore (T23/T24), never a
re-embed. That is why this module has no notion of a remote target: the artefact it
produces is rows in Postgres, and copying rows is somebody else's task.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.orm import Session

from paddock.db.models import Chunk, IncidentComment, Meeting
from paddock.db.session import session_scope
from paddock.embed.chunker import SOURCE_TYPE
from paddock.embed.embedder import Embedder
from paddock.embed.store import EmbedResult, embed_meeting
from paddock.retrieval.vector_tool import search_comments

BENCHMARK_QUERIES: tuple[str, ...] = (
    "trouble in running",
    "hampered approaching the 800 Metres",
    "slow to begin and lost ground at the start",
    "veterinary examination revealed no abnormality",
    "raced wide throughout without cover",
    "checked when awkwardly placed behind a weakening rival",
    # Chinese probes, because a Chinese question is half the workload and a
    # cross-lingual vector is no cheaper or dearer to scan than an English one. They
    # are here to make the benchmark the real mixture, not to test the model.
    "受阻",
    "起步緩慢",
)
"""The probe set. Short and varied, like the questions the agent actually embeds."""


@dataclass(frozen=True)
class MeetingOutcome:
    """What became of one meeting."""

    meeting_id: int
    race_date: dt.date
    racecourse: str
    result: EmbedResult | None = None
    error: str | None = None
    """Set when the meeting failed. Its transaction rolled back, so nothing of it
    reached the corpus and a re-run starts it from the beginning."""


@dataclass
class CorpusReport:
    outcomes: list[MeetingOutcome] = field(default_factory=list)
    orphans_deleted: int = 0
    """Chunks whose comment no longer exists, removed before the walk began."""

    @property
    def succeeded(self) -> list[MeetingOutcome]:
        return [outcome for outcome in self.outcomes if outcome.error is None]

    @property
    def failed(self) -> list[MeetingOutcome]:
        return [outcome for outcome in self.outcomes if outcome.error is not None]

    @property
    def meetings(self) -> int:
        return len(self.outcomes)

    def _sum(self, attribute: str) -> int:
        return sum(
            getattr(outcome.result, attribute) for outcome in self.succeeded if outcome.result
        )

    @property
    def comments(self) -> int:
        return self._sum("comments")

    @property
    def total(self) -> int:
        """Chunks the corpus should hold for the meetings that succeeded."""
        return self._sum("total")

    @property
    def embedded(self) -> int:
        """Newly encoded. Zero across a whole run means the corpus was already done."""
        return self._sum("embedded")

    @property
    def unchanged(self) -> int:
        return self._sum("unchanged")

    @property
    def deleted(self) -> int:
        return self._sum("deleted")


def embed_corpus(
    *,
    embedder: Embedder,
    since: dt.date | None = None,
    until: dt.date | None = None,
    on_outcome: Callable[[MeetingOutcome], None] | None = None,
) -> CorpusReport:
    """Embed every stored meeting, oldest first, one transaction each.

    Args:
        embedder: the model. The same one that embedded the rest of the corpus, or
            the vectors are not comparable to each other.
        since: on or after this race date. Defaults to the start of the corpus.
        until: on or before this race date.
        on_outcome: called with each `MeetingOutcome` as it lands. A half-hour run
            that prints nothing is indistinguishable from a hung one.
    """
    report = CorpusReport(orphans_deleted=_prune_orphans())

    for meeting_id, race_date, racecourse in _meetings(since=since, until=until):
        outcome = _one_meeting(meeting_id, race_date, racecourse, embedder=embedder)
        report.outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    return report


def _prune_orphans() -> int:
    """Delete chunks whose comment is gone. Its own transaction, before the walk.

    Not bounded by `since`/`until`: an orphan has no meeting to belong to any more,
    so a window cannot select one. Deleting all of them on any run is the only reading
    that leaves the corpus in the same state whatever window was asked for.
    """
    with session_scope() as session:
        result = cast(
            "CursorResult[Any]",
            session.execute(
                delete(Chunk).where(
                    Chunk.source_type == SOURCE_TYPE,
                    ~select(IncidentComment.id)
                    .where(IncidentComment.id == Chunk.source_id)
                    .exists(),
                )
            ),
        )
        return result.rowcount or 0


def _meetings(*, since: dt.date | None, until: dt.date | None) -> list[tuple[int, dt.date, str]]:
    """The meetings to walk, oldest first.

    Read in one query up front rather than held open across the run: the walk takes
    half an hour, and a cursor kept open for it would pin a transaction for the same
    length of time.
    """
    statement = select(Meeting.id, Meeting.race_date, Meeting.racecourse).order_by(
        Meeting.race_date, Meeting.id
    )
    if since is not None:
        statement = statement.where(Meeting.race_date >= since)
    if until is not None:
        statement = statement.where(Meeting.race_date <= until)

    with session_scope() as session:
        return [(row[0], row[1], row[2]) for row in session.execute(statement).all()]


def _one_meeting(
    meeting_id: int, race_date: dt.date, racecourse: str, *, embedder: Embedder
) -> MeetingOutcome:
    try:
        with session_scope() as session:
            result = embed_meeting(session, meeting_id=meeting_id, embedder=embedder)
    except Exception as error:
        return MeetingOutcome(
            meeting_id, race_date, racecourse, error=f"{type(error).__name__}: {error}"
        )
    return MeetingOutcome(meeting_id, race_date, racecourse, result=result)


# ── The audit ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Coverage:
    """How much of the corpus is reachable by vector search.

    `chunks` is not compared with `comments`: since ADR-003 one comment is several
    chunks, so the two counts are not meant to agree. What must hold is that every
    comment produced at least one chunk — a comment with none is invisible to
    retrieval while looking perfectly healthy in SQL.
    """

    comments: int
    with_chunks: int
    chunks: int
    orphans: int
    """Chunks citing a comment id that no longer exists. Retrievable and uncitable —
    the worse half of an incomplete corpus, because it looks like coverage."""

    @property
    def without_chunks(self) -> int:
        return self.comments - self.with_chunks

    @property
    def complete(self) -> bool:
        return self.without_chunks == 0 and self.orphans == 0


def chunk_coverage(session: Session) -> Coverage:
    """Count comments, comments that have a chunk, and chunks."""
    comments = session.scalar(select(func.count(IncidentComment.id))) or 0
    with_chunks = (
        session.scalar(
            select(func.count(func.distinct(Chunk.source_id)))
            .select_from(Chunk)
            .join(IncidentComment, IncidentComment.id == Chunk.source_id)
            .where(Chunk.source_type == SOURCE_TYPE)
        )
        or 0
    )
    chunks = (
        session.scalar(select(func.count(Chunk.id)).where(Chunk.source_type == SOURCE_TYPE)) or 0
    )
    orphans = (
        session.scalar(
            select(func.count(Chunk.id)).where(
                Chunk.source_type == SOURCE_TYPE,
                ~select(IncidentComment.id).where(IncidentComment.id == Chunk.source_id).exists(),
            )
        )
        or 0
    )
    return Coverage(comments=comments, with_chunks=with_chunks, chunks=chunks, orphans=orphans)


@dataclass(frozen=True)
class Latency:
    """Timings for the retrieval half of a question, in milliseconds."""

    samples: int
    p50_ms: float
    p95_ms: float
    max_ms: float


class _Precomputed:
    """An `Embedder` that returns vectors it was handed. Never encodes anything.

    The p95 budget in T12 is a claim about the HNSW index, not about bge-m3 on two
    CPU cores. Encoding one short query costs tens of milliseconds and would dominate
    the measurement, so every query is encoded once, before the clock starts, and the
    timed loop reads from here.
    """

    def __init__(self, vectors: dict[str, list[float]], *, dim: int) -> None:
        self._vectors = vectors
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]


def benchmark_search(
    session: Session,
    *,
    queries: Sequence[str] = BENCHMARK_QUERIES,
    embedder: Embedder,
    repeats: int = 10,
    limit: int = 5,
) -> Latency:
    """Time `search_comments` over `queries`, `repeats` times each.

    What is timed is one whole retrieval — the metadata filter, the ANN scan and the
    keyed fetch of each hit's remaining sentences — because that is what a question
    waits for. What is not timed is encoding the question; see `_Precomputed`.

    Raises:
        ValueError: `queries` is empty, or `repeats` is below 1.
    """
    if not queries:
        raise ValueError("benchmark needs at least one query")
    if repeats < 1:
        raise ValueError(f"repeats must be 1 or more, got {repeats}")

    vectors = dict(zip(queries, embedder.embed(list(queries)), strict=True))
    cached = _Precomputed(vectors, dim=embedder.dim)

    samples: list[float] = []
    for _ in range(repeats):
        for query in queries:
            start = time.perf_counter()
            search_comments(session, query=query, embedder=cached, limit=limit)
            samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    return Latency(
        samples=len(samples),
        p50_ms=_percentile(samples, 0.50),
        p95_ms=_percentile(samples, 0.95),
        max_ms=samples[-1],
    )


def _percentile(sorted_samples: list[float], fraction: float) -> float:
    """Nearest-rank percentile. No interpolation — these are latencies, not a curve."""
    index = round(fraction * (len(sorted_samples) - 1))
    return sorted_samples[index]


__all__ = [
    "BENCHMARK_QUERIES",
    "CorpusReport",
    "Coverage",
    "Latency",
    "MeetingOutcome",
    "benchmark_search",
    "chunk_coverage",
    "embed_corpus",
]
