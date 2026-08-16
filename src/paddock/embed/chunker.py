"""Turning incident comments into chunks.

## One comment, one chunk, no splitting

Stewards' comments are one to four sentences and already semantically atomic. A
splitter would sever "was hampered at the 800" from the horse it happened to, which
is the only fact in the sentence worth retrieving. So the chunk boundary is the
comment boundary, and there is no chunk-size parameter to tune.

## The text is the narrative, and nothing else

It is tempting to prepend "SETANTA, Race 5 at Sha Tin over 1200m:" to every chunk so
that a bare name query matches. We do not, because retrieval here is
metadata-*pre*-filtered: the identifying facts live in ``chunk_meta`` and constrain
the ANN scan inside the same SQL statement (spec §4). Embedding them again would
spend the vector's capacity on facts the filter already handles exactly, and would
pull every chunk of a busy meeting closer together — the horse names would dominate
the incidents.

## Two things called "course"

``racecourse`` is the venue, ST or HV. ``course`` is the rail configuration — A, B,
C+3 — which moves the running rail out to spread the wear on the turf and changes
the effective distance. The spec listed one field named ``course``; naming them apart
here is the same lesson that ``carried_weight_lb`` vs ``declared_horse_weight_lb``
taught in T6a.

## Metadata is present-and-null, never absent

Every chunk carries every key. A key that vanishes when its value is unknown turns
each metadata filter into two code paths — one for the JSONB match and one for the
rows where the key never existed — and the second path is the one nobody writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from paddock.db.models import IncidentComment, Meeting, Race

SOURCE_TYPE = "incident_comment"


@dataclass(frozen=True)
class ChunkSpec:
    """What to embed and what to file it under.

    Deliberately not a ``Chunk`` ORM row: this is the pure description of a chunk,
    built without a session, and the embedding is attached later. The store turns it
    into a row.
    """

    source_type: str
    source_id: int
    text: str
    chunk_meta: dict[str, Any]


def build_chunk(comment: IncidentComment, *, race: Race, meeting: Meeting) -> ChunkSpec:
    """Describe the chunk for one incident comment.

    Args:
        comment: the stewards' narrative about one runner.
        race: the race it was made in — supplies distance, class and rail position.
        meeting: that race's meeting — supplies the date and the venue.

    Raises:
        ValueError: the comment is empty, or does not belong to `race`.
    """
    if comment.race_id != race.id:
        raise ValueError(
            f"comment {comment.id} belongs to race {comment.race_id}, not race {race.id}"
        )

    text = _collapse_whitespace(comment.text_en)
    if not text:
        raise ValueError(f"comment {comment.id} has empty text — nothing to embed")

    return ChunkSpec(
        source_type=SOURCE_TYPE,
        source_id=comment.id,
        text=text,
        chunk_meta={
            "horse_id": comment.horse_id,
            "race_id": race.id,
            "race_no": race.race_no,
            # JSONB has no date type. ISO-8601 sorts lexicographically, so a range
            # filter works on the string without casting it back to a date.
            "race_date": meeting.race_date.isoformat(),
            "racecourse": meeting.racecourse,
            "course": race.course,
            "distance_m": race.distance_m,
            "race_class": race.race_class,
            # Going is recorded for the meeting but is revised during it — a track
            # downgraded after rain applies from that race on, so the race's own
            # value is the truthful one wherever it exists.
            "going": race.going or meeting.going,
            "finish_pos": comment.finish_pos,
        },
    )


def _collapse_whitespace(text: str | None) -> str:
    """Comment cells arrive with newlines and runs of spaces from the HTML."""
    return re.sub(r"\s+", " ", text or "").strip()
