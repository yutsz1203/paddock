"""Semantic search over stewards' comments — filtered before the scan, not after.

## Why the filter has to be inside the statement

The obvious implementation is: embed the question, take the 20 nearest comments,
then drop the ones that are not about this horse. It works on a demo corpus and
fails on a real one. A horse has perhaps twenty comments among 12,000; ask "did
SETANTA have trouble in running last time?" and the twenty globally nearest comments
about interference will be about twenty *other* horses, all of which were also
hampered. Post-filtering returns nothing, from a corpus that contains the answer.

So the metadata predicate and the ANN ordering go into one statement, and Postgres
applies the predicate first. That is the entire reason `chunk_meta` duplicates
columns that already exist relationally (spec §4) — pgvector can filter and scan in
one query, which is what makes hybrid retrieval a query rather than a reranking
problem.

## Equality filters use `@>`, ranges use `->>`

The GIN index on `chunk_meta` is a containment index: ``chunk_meta @> '{"horse_id":
"HK_2024_K570"}'`` uses it, ``chunk_meta ->> 'horse_id' = '…'`` does not. Equality
filters therefore go through `.contains()`. A date window cannot be expressed as
containment, so it compares the ISO string with `->>` — which is why the chunker
stores dates as ISO-8601 in the first place: the text comparison and the date
comparison agree.

## The unit stored is a sentence; the unit returned is a comment, with all its pieces

Since Checkpoint A a comment is embedded one sentence-sized piece at a time
(ADR-003), so the nearest five *chunks* can be five pieces of one long comment —
one citation, dressed as five. `search_comments` therefore over-fetches by
`_FANOUT` and returns `limit` distinct *comments*, ranked by their nearest piece.

But a hit is not that one piece. **Sentences are the unit of finding; the comment is
the unit of evidence.** The dilution ADR-003 removed was a property of the *index* —
one vector averaging eight facts into a centroid near none of them. Nothing equivalent
happens to a reader: a model handed 1000 characters of stewards' report is not
averaging it, and the cost of the extra sentences is tokens, not correctness. Cutting
finely to search and then quoting whole is therefore not a compromise between the two,
it is each granularity used where it works.

Checkpoint A paid for that sentence twice. Keeping only the nearest piece handed the
model the stewards' *sanction* and dropped "delayed the start when it proved difficult
to load"; adding a per-piece distance rule then dropped the jockey's account of the
run at 0.607, seven thousandths outside a threshold, while the same threshold admitted
a negative control at 0.374 in another query. Both answers were fluent, cited, and
wrong. So a hit carries **every** piece of its comment, in reading order, and the
distances travel with them for tracing rather than for filtering.

The pieces come from a second, keyed lookup rather than from the ANN window. A window
wide enough to rank comments is not wide enough to hold all of them, and "some of the
evidence, silently" is the bug this paragraph exists to prevent.

The collapse happens in Python rather than as a `DISTINCT ON (source_id)` because
the SQL form has to sort the whole filtered set by source before it can dedupe,
which is exactly the full scan the HNSW index exists to avoid. Over-fetching keeps
the ANN ordering doing the work.

## No caller supplies SQL

`search_comments` takes typed keyword arguments and nothing else. Filter values are
bound parameters, so a horse id that arrived from a user prompt is data. The agent
selects this tool and fills its arguments; it never composes a query.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Row, Select, select
from sqlalchemy.orm import Session

from paddock.db.models import Chunk
from paddock.embed.chunker import SOURCE_TYPE
from paddock.embed.embedder import Embedder

MAX_LIMIT = 50
"""Ceiling on one call, so a miscounted `limit` cannot pull the corpus into a prompt."""

_FANOUT = 4
"""Chunks scanned per requested comment, so that `limit` distinct comments come back.

The seeded meeting averages 1.9 chunks per comment and its longest holds 7, so 4
covers the common case without reading a page of vectors to rank five comments. It is
a recall/latency trade and not a correctness one: too low returns fewer than `limit`
comments, never a wrong one, and never a partial one — the pieces of a comment that
does come back are fetched by key, not taken from this window.
"""


@dataclass(frozen=True)
class CommentPiece:
    """One sentence-sized chunk of a comment, and how near the question it sits."""

    chunk_index: int
    """Position within the comment, from 0 — reading order."""
    text: str
    distance: float
    """Cosine distance, 0 (identical) to 2. Lower is nearer."""


@dataclass(frozen=True)
class CommentHit:
    """One retrieved comment, carrying everything a citation needs.

    The metadata travels with the hit so a source card renders without a second
    round trip: an answer whose citation cannot be resolved is not a citation.
    """

    comment_id: int
    """`incident_comments.id` — the row these pieces were made from."""
    pieces: list[CommentPiece]
    """Every piece of the comment, in reading order — the whole of it, always."""
    distance: float
    """The nearest piece's distance — what this comment was ranked by."""
    horse_id: str
    race_id: int
    race_no: int
    race_date: dt.date
    racecourse: str
    distance_m: int | None
    race_class: str | None
    going: str | None
    finish_pos: int | None

    @property
    def nearest(self) -> CommentPiece:
        """The piece this comment was ranked by.

        For tracing, and for a UI that wants to highlight what matched — never for
        deciding what to show. Selecting evidence by per-piece distance is what
        Checkpoint A measured and rejected: admitting a comment and choosing the
        sentences inside it are different questions, and the second has no answer that
        is not fitted to one query.
        """
        return min(self.pieces, key=lambda piece: piece.distance)

    @property
    def text(self) -> str:
        """The comment as the stewards wrote it — every piece, in reading order."""
        return " ".join(piece.text for piece in self.pieces)


def search_comments(
    session: Session,
    *,
    query: str,
    embedder: Embedder,
    horse_id: str | None = None,
    racecourse: str | None = None,
    distance_m: int | None = None,
    since: dt.date | None = None,
    until: dt.date | None = None,
    limit: int = 5,
) -> list[CommentHit]:
    """The comments nearest to `query`, among those matching every filter given.

    One hit per comment, ranked by its nearest sentence, and each hit carries every
    sentence of that comment with its own distance. Several sentences matching is one
    source that says several things — not several sources, and not one sentence.

    Args:
        session: an open session.
        query: the question, in any language bge-m3 covers.
        embedder: the model to embed `query` with — the same one that embedded the
            corpus, or the vectors are not comparable.
        horse_id: restrict to one horse. This is the filter that matters: without it
            a question about one horse retrieves other horses' incidents.
        racecourse: ST or HV.
        distance_m: exact race distance.
        since: on or after this date.
        until: on or before this date.
        limit: how many comments, 1..`MAX_LIMIT`. Fewer come back when the nearest
            chunks cluster inside a handful of comments — see `_FANOUT`.

    Raises:
        ValueError: the query is empty, or `limit` is outside 1..`MAX_LIMIT`.
    """
    if not query.strip():
        raise ValueError("query is empty — an empty vector is weakly near everything")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}")

    vector = embedder.embed([query])[0]
    distance = Chunk.embedding.cosine_distance(vector)

    statement = _filtered(
        select(Chunk, distance.label("distance")),
        horse_id=horse_id,
        racecourse=racecourse,
        distance_m=distance_m,
        since=since,
        until=until,
    ).order_by(distance)

    rows = session.execute(statement.limit(limit * _FANOUT)).all()
    comment_ids = _rank_comments(rows, limit=limit)
    if not comment_ids:
        return []
    return _assemble(session, comment_ids, distance=distance)


def _rank_comments(rows: Sequence[Row[Any]], *, limit: int) -> list[int]:
    """The comments in the ANN window, best first, at most `limit` of them.

    The rows arrive sorted, so a comment's first appearance is its nearest piece and
    fixes its rank. Later appearances say nothing new about where it belongs.
    """
    ranked: list[int] = []
    seen: set[int] = set()
    for chunk, _ in rows:
        if chunk.source_id in seen:
            continue
        seen.add(chunk.source_id)
        ranked.append(chunk.source_id)
        if len(ranked) == limit:
            break
    return ranked


def _assemble(session: Session, comment_ids: list[int], *, distance: Any) -> list[CommentHit]:
    """Fetch every piece of the ranked comments and build one hit each.

    Keyed on `source_id`, so a hit holds all of its comment however wide the ANN
    window was. The distance is recomputed here for the pieces the window did not
    reach — the same expression against a handful of rows, not a second scan.
    """
    rows = session.execute(
        select(Chunk, distance.label("distance"))
        .where(Chunk.source_type == SOURCE_TYPE, Chunk.source_id.in_(comment_ids))
        .order_by(Chunk.source_id, Chunk.chunk_index)
    ).all()

    grouped: dict[int, list[tuple[Chunk, float]]] = {}
    for chunk, distance_value in rows:
        grouped.setdefault(chunk.source_id, []).append((chunk, float(distance_value)))

    # Ranked order, not `grouped` order: the ranking came from the ANN scan.
    return [_hit(grouped[comment_id]) for comment_id in comment_ids if comment_id in grouped]


def _filtered(
    statement: Select[Any],
    *,
    horse_id: str | None,
    racecourse: str | None,
    distance_m: int | None,
    since: dt.date | None,
    until: dt.date | None,
) -> Select[Any]:
    """Attach every metadata predicate to the same statement as the ANN ordering."""
    statement = statement.where(Chunk.source_type == SOURCE_TYPE)

    # Containment (`@>`) so the GIN index on chunk_meta is usable. One call per
    # filter rather than one merged dict: a merged `@>` matches only rows carrying
    # all keys, which is the same thing here but stops being so the moment a key
    # becomes optional.
    for key, value in (
        ("horse_id", horse_id),
        ("racecourse", racecourse),
        ("distance_m", distance_m),
    ):
        if value is not None:
            statement = statement.where(Chunk.chunk_meta.contains({key: value}))

    # ISO-8601 dates compare correctly as text, which is why the chunker stores them
    # that way — containment cannot express a range.
    if since is not None:
        statement = statement.where(Chunk.chunk_meta["race_date"].astext >= since.isoformat())
    if until is not None:
        statement = statement.where(Chunk.chunk_meta["race_date"].astext <= until.isoformat())

    return statement


def _hit(rows: list[tuple[Chunk, float]]) -> CommentHit:
    """One comment's pieces, in reading order, as a hit.

    Every piece carries the same `chunk_meta` (the chunker copies it per piece), so
    the first is as good as any for the metadata a source card needs.
    """
    pieces = [
        CommentPiece(chunk_index=chunk.chunk_index, text=chunk.text, distance=distance)
        for chunk, distance in sorted(rows, key=lambda row: row[0].chunk_index)
    ]
    chunk = rows[0][0]
    meta = chunk.chunk_meta
    return CommentHit(
        comment_id=chunk.source_id,
        pieces=pieces,
        distance=min(piece.distance for piece in pieces),
        horse_id=meta["horse_id"],
        race_id=meta["race_id"],
        race_no=meta["race_no"],
        race_date=dt.date.fromisoformat(meta["race_date"]),
        racecourse=meta["racecourse"],
        distance_m=meta.get("distance_m"),
        race_class=meta.get("race_class"),
        going=meta.get("going"),
        finish_pos=meta.get("finish_pos"),
    )
