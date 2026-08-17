"""Turning incident comments into chunks.

## One chunk per sentence, not one chunk per comment

The first version of this file embedded each comment whole, on the premise that
stewards' comments are "one to four sentences and already semantically atomic".
Checkpoint A measured that premise and it is false. Across ten unfiltered questions
only 32 of 114 seeded comments ever entered a top 5; the reachable ones averaged 151
characters and the unreachable ones 228, and the corpus runs to 1001. Comment 724 is
one vector covering difficult loading, a jockey's explanation, top weight, kickback,
an unacceptable-performance finding, a barrier-trial order, a vet inspection and
post-race sampling — a centroid near none of its eight facts, and invisible to all
eight questions. Averaging is what a long comment does to its own topics.

So the chunk boundary is the sentence boundary, which in this corpus is the topic
boundary: HKJC writes one incident, one finding or one explanation per sentence.

## Splitting is safe because the horse is not in the text

The fear that stopped the first version was that splitting severs "was hampered at
the 800" from the horse it happened to. It does not: `horse_id` lives in
`chunk_meta`, retrieval is metadata-*pre*-filtered (spec §4), and the citation still
addresses the whole comment — `source_id` is the comment id whatever the sentence.
The decision that makes splitting safe was already two paragraphs above the decision
not to split.

## A fragment is merged, a long sentence is not broken

Two bounds keep the pieces useful. Anything under `MIN_CHARS` ("Sent for sampling
post-race.") joins its neighbour rather than becoming a vector of its own, because a
fragment that short is near everything weakly. Nothing splits *inside* a sentence:
the longest in the seeded meeting is 354 characters, still well under the length at
which dilution started to bite, and a mid-sentence cut would strip the clause that
says who it happened to.

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

MIN_CHARS = 40
"""Below this a piece is a fragment, and joins its neighbour instead of standing alone."""

# A sentence ends at a terminator followed by whitespace and something that opens a
# sentence — a capital, a quote, a bracket, or the '<' of HKJC's
# "<27/4/2026 Additional Veterinary Report>" sub-headers. Requiring the opener is what
# keeps "1.5 lengths" and "$5,000." intact, and HK stewards write initials without
# stops ("K C Leung"), so the usual abbreviation trap does not apply here. Verified
# against all 114 comments of the seeded meeting: no false boundary.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“<(\[])')


@dataclass(frozen=True)
class ChunkSpec:
    """What to embed and what to file it under.

    Deliberately not a ``Chunk`` ORM row: this is the pure description of a chunk,
    built without a session, and the embedding is attached later. The store turns it
    into a row.
    """

    source_type: str
    source_id: int
    chunk_index: int
    """Position within the comment, from 0. With `source_id` it addresses the chunk."""
    text: str
    chunk_meta: dict[str, Any]


def build_chunks(comment: IncidentComment, *, race: Race, meeting: Meeting) -> list[ChunkSpec]:
    """Describe the chunks for one incident comment, in reading order.

    Args:
        comment: the stewards' narrative about one runner.
        race: the race it was made in — supplies distance, class and rail position.
        meeting: that race's meeting — supplies the date and the venue.

    Returns:
        One spec per sentence-sized piece; a short comment yields exactly one. Every
        spec carries the same metadata and the same `source_id`, so a citation
        resolves to the comment however the text was cut.

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

    chunk_meta = {
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
    }

    return [
        ChunkSpec(
            source_type=SOURCE_TYPE,
            source_id=comment.id,
            chunk_index=index,
            text=piece,
            # A copy per chunk: they are equal today, and a shared dict would make a
            # later per-chunk key (a section marker, say) silently global.
            chunk_meta=dict(chunk_meta),
        )
        for index, piece in enumerate(split_sentences(text))
    ]


def split_sentences(text: str) -> list[str]:
    """Cut `text` at sentence boundaries, folding fragments into their neighbour.

    A piece shorter than `MIN_CHARS` is joined to the piece before it — or, if it is
    the first, the piece after it arrives and joins it. The result is that no chunk
    is a stub, and a comment that is itself a stub ("Slow to begin.") stays whole
    rather than being padded with an unrelated sentence.
    """
    pieces: list[str] = []
    for part in (part.strip() for part in _SENTENCE_BOUNDARY.split(text)):
        if not part:
            continue
        if pieces and (len(part) < MIN_CHARS or len(pieces[-1]) < MIN_CHARS):
            pieces[-1] = f"{pieces[-1]} {part}"
        else:
            pieces.append(part)
    return pieces


def _collapse_whitespace(text: str | None) -> str:
    """Comment cells arrive with newlines and runs of spaces from the HTML."""
    return re.sub(r"\s+", " ", text or "").strip()
