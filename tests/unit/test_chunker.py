"""Turning an incident comment into a chunk — no database, no model.

The two things worth testing here are both decisions rather than mechanics:

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
from paddock.embed.chunker import build_chunk

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


def test_chunk_text_is_the_comment_verbatim() -> None:
    chunk = build_chunk(_comment(), race=_race(), meeting=_meeting())

    assert chunk.text == "Was hampered approaching the 800 Metres and lost ground."


def test_chunk_is_keyed_to_the_comment_it_came_from() -> None:
    """One chunk per comment, addressed by source — that pair is what makes
    re-embedding idempotent rather than duplicating."""
    chunk = build_chunk(_comment(id=77), race=_race(), meeting=_meeting())

    assert chunk.source_type == "incident_comment"
    assert chunk.source_id == 77


def test_metadata_carries_every_field_retrieval_filters_on() -> None:
    chunk = build_chunk(_comment(), race=_race(), meeting=_meeting())

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
    chunk = build_chunk(_comment(), race=_race(), meeting=_meeting())

    assert chunk.chunk_meta["race_date"] == "2026-04-26"


def test_race_going_wins_over_meeting_going() -> None:
    """Going is recorded per meeting but changes during it — a track downgraded after
    rain applies from that race on, so the race's own value is the truthful one."""
    chunk = build_chunk(
        _comment(),
        race=_race(going="YIELDING"),
        meeting=_meeting(going="GOOD"),
    )

    assert chunk.chunk_meta["going"] == "YIELDING"


def test_missing_facts_are_null_not_absent() -> None:
    """A key that disappears when its value is unknown makes every metadata filter
    conditional. Present-and-null is one filter; absent is two code paths."""
    chunk = build_chunk(
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
    chunk = build_chunk(
        _comment(finish_pos=None, text_en="Pulled up before the 400 Metres."),
        race=_race(),
        meeting=_meeting(),
    )

    assert chunk.chunk_meta["finish_pos"] is None
    assert chunk.text == "Pulled up before the 400 Metres."


def test_whitespace_is_collapsed() -> None:
    """Comment cells arrive with newlines and runs of spaces from the HTML. The
    embedding is unbothered, but the citation shown to a reader is not."""
    chunk = build_chunk(
        _comment(text_en="  Was hampered\n   approaching the 800 Metres.  "),
        race=_race(),
        meeting=_meeting(),
    )

    assert chunk.text == "Was hampered approaching the 800 Metres."


def test_an_empty_comment_is_refused() -> None:
    """ "No report." is stored as no comment at all (T4), so an empty text here means
    something upstream is wrong. Embedding it would put an empty vector in the corpus
    that matches everything weakly."""
    with pytest.raises(ValueError, match="empty"):
        build_chunk(_comment(text_en="   "), race=_race(), meeting=_meeting())


def test_the_comment_must_belong_to_the_race() -> None:
    """A mismatched pair would attach a comment to the wrong race's metadata — the
    kind of silent corruption that only shows up as a wrong citation months later."""
    with pytest.raises(ValueError, match="race"):
        build_chunk(_comment(race_id=10), race=_race(id=11), meeting=_meeting())
