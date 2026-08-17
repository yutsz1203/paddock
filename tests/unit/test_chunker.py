"""Turning an incident comment into chunks — no database, no model.

The things worth testing here are decisions rather than mechanics:

**A comment is cut at its sentences.** Checkpoint A measured one-vector-per-comment
and found 72% of the corpus unreachable, with length predicting retrievability almost
perfectly. So the tests below pin the boundary rule, the fragment merge, and the two
invariants that make splitting safe: every piece keeps the comment's `source_id`, and
every piece carries the whole metadata.

**The chunk text is the stewards' narrative verbatim.** No horse name, no race
header, no summarisation. Retrieval is metadata-pre-filtered, so the identifying
facts are handled by the filter; putting them in the embedded text would dilute the
one thing the vector is for — what happened during the race.

**Metadata names two different "courses".** ``racecourse`` is the venue (ST or HV);
``course`` is the rail configuration (A, B, C+3). They were one field in the spec,
which is exactly the confusion that ``carried_weight_lb`` vs
``declared_horse_weight_lb`` already cost us once.
"""

from __future__ import annotations

import datetime as dt

import pytest

from paddock.db.models import IncidentComment, Meeting, Race
from paddock.embed.chunker import ChunkSpec, build_chunks, split_sentences

RACE_DATE = dt.date(2026, 4, 26)


def _meeting(**overrides: object) -> Meeting:
    fields: dict[str, object] = {
        "id": 1,
        "race_date": RACE_DATE,
        "racecourse": "ST",
        "going": "GOOD",
    }
    return Meeting(**(fields | overrides))


def _race(**overrides: object) -> Race:
    fields: dict[str, object] = {
        "id": 10,
        "meeting_id": 1,
        "race_no": 5,
        "race_class": "Class 4",
        "distance_m": 1200,
        "course": "A",
        "going": None,
        "status": "finished",
    }
    return Race(**(fields | overrides))


def _comment(**overrides: object) -> IncidentComment:
    fields: dict[str, object] = {
        "id": 77,
        "race_id": 10,
        "horse_id": "HK_2024_K570",
        "finish_pos": 4,
        "text_en": "Was hampered approaching the 800 Metres and lost ground.",
    }
    return IncidentComment(**(fields | overrides))


def _only(
    comment: IncidentComment, *, race: Race | None = None, meeting: Meeting | None = None
) -> ChunkSpec:
    """The single chunk of a one-sentence comment — most tests below want just it."""
    chunks = build_chunks(comment, race=race or _race(), meeting=meeting or _meeting())
    assert len(chunks) == 1
    return chunks[0]


def test_chunk_text_is_the_comment_verbatim() -> None:
    chunk = build_chunks(_comment(), race=_race(), meeting=_meeting())[0]

    assert chunk.text == "Was hampered approaching the 800 Metres and lost ground."


def test_a_short_comment_stays_one_chunk() -> None:
    """Half the corpus is a single sentence. Splitting must not manufacture pieces
    that were never there."""
    chunks = build_chunks(_comment(), race=_race(), meeting=_meeting())

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_a_long_comment_is_cut_at_its_sentences() -> None:
    """The Checkpoint A finding, as a test: comment 724 was 1001 characters covering
    eight unrelated facts in one vector, and was invisible to all eight questions."""
    chunks = build_chunks(
        _comment(
            text_en=(
                "Delayed the start when it proved difficult to load. "
                "K C Leung stated that his mount jumped only fairly and had to be "
                "ridden along in the early stages to hold its position. "
                "A veterinary inspection immediately following the race did not show "
                "any significant findings."
            )
        ),
        race=_race(),
        meeting=_meeting(),
    )

    assert [chunk.text for chunk in chunks] == [
        "Delayed the start when it proved difficult to load.",
        "K C Leung stated that his mount jumped only fairly and had to be ridden along "
        "in the early stages to hold its position.",
        "A veterinary inspection immediately following the race did not show any "
        "significant findings.",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_every_chunk_is_keyed_to_the_comment_it_came_from() -> None:
    """Splitting is only safe because the citation does not split with it: whichever
    sentence matched, the source resolves to the whole comment."""
    chunks = build_chunks(
        _comment(
            id=77,
            text_en=(
                "Jumped awkwardly and lost several lengths at the start. "
                "When questioned the rider stated that his mount was never travelling "
                "and failed to respond to pressure in the Home Straight."
            ),
        ),
        race=_race(),
        meeting=_meeting(),
    )

    assert len(chunks) == 2
    assert {chunk.source_type for chunk in chunks} == {"incident_comment"}
    assert {chunk.source_id for chunk in chunks} == {77}
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_every_chunk_carries_the_full_metadata() -> None:
    """The filter runs against the chunk, not the comment. A second piece with thinner
    metadata would be unreachable by exactly the queries that scope to a horse."""
    chunks = build_chunks(
        _comment(
            text_en=(
                "Jumped awkwardly and lost several lengths at the start. "
                "When questioned the rider stated that his mount was never travelling "
                "and failed to respond to pressure in the Home Straight."
            )
        ),
        race=_race(),
        meeting=_meeting(),
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_meta == chunks[1].chunk_meta
    assert chunks[1].chunk_meta["horse_id"] == "HK_2024_K570"


def test_a_fragment_joins_its_neighbour_rather_than_standing_alone() -> None:
    """ "Sent for sampling post-race." is 28 characters and appears verbatim on dozens
    of runners. As its own vector it is near every question weakly and no question
    strongly — Checkpoint A watched three copies of it answer a trainer question."""
    chunks = build_chunks(
        _comment(
            text_en=(
                "A veterinary inspection immediately following the race did not show "
                "any significant findings. Sent for sampling post-race."
            )
        ),
        race=_race(),
        meeting=_meeting(),
    )

    assert len(chunks) == 1
    assert chunks[0].text.endswith("Sent for sampling post-race.")


def test_a_sentence_is_never_cut_in_half() -> None:
    """There is no maximum chunk size. A 350-character sentence is one topic stated at
    length, and cutting it would strip the clause naming who it happened to."""
    long_sentence = (
        "M L Yeung pleaded guilty to a charge of careless riding in that near the 250 "
        "Metres he allowed his mount to shift out when not clear of AESTHETICISM, "
        "resulting in AESTHETICISM being badly crowded and taken out onto TYCOON "
        "EXPRESS and also contributing to TYCOON EXPRESS becoming badly crowded to "
        "the inside of MASSIVE GLORY."
    )

    assert split_sentences(long_sentence) == [long_sentence]


def test_a_decimal_point_is_not_a_sentence_boundary() -> None:
    """Stewards write margins with a stop inside them. A boundary needs a terminator,
    whitespace *and* something that opens a sentence — which is what keeps this whole
    where a naive split on "." would produce "Was beaten 1" and "5 lengths…"."""
    text = "Was beaten 1.5 lengths after being crowded near the 400 Metres."

    assert split_sentences(text) == [text]


def test_metadata_carries_every_field_retrieval_filters_on() -> None:
    chunk = _only(_comment())

    assert chunk.chunk_meta == {
        "horse_id": "HK_2024_K570",
        "race_id": 10,
        "race_no": 5,
        "race_date": "2026-04-26",
        "racecourse": "ST",
        "course": "A",
        "distance_m": 1200,
        "race_class": "Class 4",
        "going": "GOOD",
        "finish_pos": 4,
    }


def test_race_date_is_an_iso_string() -> None:
    """JSONB has no date type. ISO-8601 sorts lexicographically, so a range filter
    on the metadata still works without casting."""
    chunk = _only(_comment())

    assert chunk.chunk_meta["race_date"] == "2026-04-26"


def test_race_going_wins_over_meeting_going() -> None:
    """Going is recorded per meeting but changes during it — a track downgraded after
    rain applies from that race on, so the race's own value is the truthful one."""
    chunk = _only(_comment(), race=_race(going="YIELDING"), meeting=_meeting(going="GOOD"))

    assert chunk.chunk_meta["going"] == "YIELDING"


def test_missing_facts_are_null_not_absent() -> None:
    """A key that disappears when its value is unknown makes every metadata filter
    conditional. Present-and-null is one filter; absent is two code paths."""
    chunk = _only(
        _comment(finish_pos=None),
        race=_race(distance_m=None, race_class=None, course=None, going=None),
        meeting=_meeting(going=None),
    )

    assert chunk.chunk_meta["distance_m"] is None
    assert chunk.chunk_meta["race_class"] is None
    assert chunk.chunk_meta["course"] is None
    assert chunk.chunk_meta["going"] is None
    assert chunk.chunk_meta["finish_pos"] is None


def test_a_dnf_runner_still_chunks() -> None:
    """No finishing position often means the horse broke down or was pulled up —
    the comments that matter most for form. Dropping them would be the worst
    possible filter."""
    chunk = _only(_comment(finish_pos=None, text_en="Pulled up before the 400 Metres."))

    assert chunk.chunk_meta["finish_pos"] is None
    assert chunk.text == "Pulled up before the 400 Metres."


def test_whitespace_is_collapsed() -> None:
    """Comment cells arrive with newlines and runs of spaces from the HTML. The
    embedding is unbothered, but the citation shown to a reader is not."""
    chunk = _only(_comment(text_en="  Was hampered\n   approaching the 800 Metres.  "))

    assert chunk.text == "Was hampered approaching the 800 Metres."


def test_an_empty_comment_is_refused() -> None:
    """ "No report." is stored as no comment at all (T4), so an empty text here means
    something upstream is wrong. Embedding it would put an empty vector in the corpus
    that matches everything weakly."""
    with pytest.raises(ValueError, match="empty"):
        build_chunks(_comment(text_en="   "), race=_race(), meeting=_meeting())


def test_the_comment_must_belong_to_the_race() -> None:
    """A mismatched pair would attach a comment to the wrong race's metadata — the
    kind of silent corruption that only shows up as a wrong citation months later."""
    with pytest.raises(ValueError, match="race"):
        build_chunks(_comment(race_id=10), race=_race(id=11), meeting=_meeting())
