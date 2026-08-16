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

## No caller supplies SQL

`search_comments` takes typed keyword arguments and nothing else. Filter values are
bound parameters, so a horse id that arrived from a user prompt is data. The agent
selects this tool and fills its arguments; it never composes a query.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from paddock.db.models import Chunk
from paddock.embed.chunker import SOURCE_TYPE
from paddock.embed.embedder import Embedder

MAX_LIMIT = 50
"""Ceiling on one call, so a miscounted `limit` cannot pull the corpus into a prompt."""


@dataclass(frozen=True)
class CommentHit:
    """One retrieved comment, carrying everything a citation needs.

    The metadata travels with the hit so a source card renders without a second
    round trip: an answer whose citation cannot be resolved is not a citation.
    """

    comment_id: int
    """`incident_comments.id` — the row this chunk was made from."""
    text: str
    distance: float
    """Cosine distance, 0 (identical) to 2. Lower is nearer."""
    horse_id: str
    race_id: int
    race_no: int
    race_date: dt.date
    racecourse: str
    distance_m: int | None
    race_class: str | None
    going: str | None
    finish_pos: int | None


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
        limit: how many hits, 1..`MAX_LIMIT`.

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

    rows = session.execute(statement.limit(limit)).all()
    return [_hit(chunk, float(distance_value)) for chunk, distance_value in rows]


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


def _hit(chunk: Chunk, distance: float) -> CommentHit:
    meta = chunk.chunk_meta
    return CommentHit(
        comment_id=chunk.source_id,
        text=chunk.text,
        distance=distance,
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
