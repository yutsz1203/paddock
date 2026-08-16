"""`POST /ask` over SSE.

The API is the product (spec §3): the Streamlit UI is one client among possible
many, so the contract tested here is the one that matters — events, order, and what
a client can render from them without a second request.

One deliberate property is worth stating because it looks like a bug. **Nothing is
streamed until the answer has passed the citation check.** Enforcement needs the
whole answer, and spec §10 says no uncited claim, ever — so streaming as the model
writes would mean publishing text we might have to retract. Time to first token
therefore includes generation. That is a real cost, taken knowingly; the upgrade is
to enforce per sentence and stream at that granularity, which is a T16 question, not
a T9 one.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from tests.doubles import ScriptedLLM

from paddock.agent.citations import NO_EVIDENCE
from paddock.api.main import app, get_embedder_dependency, get_llm_dependency
from paddock.db.models import EMBEDDING_DIM, Chunk, Horse, IncidentComment, Meeting, Race, Runner
from paddock.db.session import session_scope
from paddock.embed.store import embed_meeting

pytestmark = pytest.mark.integration

RACE_DATE = dt.date(2098, 12, 7)
HORSE_ID = "HK_2099_Z001"
NAME_EN = "TESTBRED FLYER"
TROUBLE = "Was hampered approaching the 800 Metres and lost ground."
CITED = "It was hampered near the 800 [S1]."


def _axis_vector(similarity: float) -> list[float]:
    return [similarity, math.sqrt(max(0.0, 1.0 - similarity**2))] + [0.0] * (EMBEDDING_DIM - 2)


class StubEmbedder:
    dim = EMBEDDING_DIM

    def __init__(self, placements: dict[str, float]) -> None:
        self.placements = placements

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_axis_vector(self.placements.get(text, 1.0)) for text in texts]


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_test_data()
    yield
    _delete_test_data()
    app.dependency_overrides.clear()


def _delete_test_data() -> None:
    with session_scope() as session:
        meeting_ids = list(
            session.scalars(select(Meeting.id).where(Meeting.race_date == RACE_DATE))
        )
        race_ids = list(session.scalars(select(Race.id).where(Race.meeting_id.in_(meeting_ids))))
        comment_ids = list(
            session.scalars(select(IncidentComment.id).where(IncidentComment.race_id.in_(race_ids)))
        )
        if comment_ids:
            session.query(Chunk).filter(Chunk.source_id.in_(comment_ids)).delete(
                synchronize_session=False
            )
        if meeting_ids:
            session.query(Meeting).filter(Meeting.id.in_(meeting_ids)).delete(
                synchronize_session=False
            )
        session.query(Horse).filter(Horse.horse_id.like("HK_2099_%")).delete(
            synchronize_session=False
        )


def _seed_and_override(*replies: str) -> ScriptedLLM:
    with session_scope() as session:
        session.add(Horse(horse_id=HORSE_ID, brand_no="Z001", name_en=NAME_EN))
        meeting = Meeting(race_date=RACE_DATE, racecourse="ST", going="GOOD")
        session.add(meeting)
        session.flush()
        race = Race(
            meeting_id=meeting.id,
            race_no=5,
            race_class="Class 4",
            distance_m=1200,
            status="finished",
        )
        session.add(race)
        session.flush()
        session.add(Runner(race_id=race.id, horse_id=HORSE_ID, horse_no=1, finish_pos=6))
        session.add(
            IncidentComment(race_id=race.id, horse_id=HORSE_ID, finish_pos=6, text_en=TROUBLE)
        )
        meeting_id = meeting.id

    embedder = StubEmbedder({TROUBLE: 0.95})
    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=embedder)

    llm = ScriptedLLM(*replies)
    app.dependency_overrides[get_llm_dependency] = lambda: llm
    app.dependency_overrides[get_embedder_dependency] = lambda: embedder
    return llm


def _events(body: str) -> list[tuple[str, dict[str, object]]]:
    """Parse an SSE body into (event, data) pairs."""
    parsed = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            parsed.append((lines["event"], json.loads(lines.get("data", "{}"))))
    return parsed


# ── The contract ────────────────────────────────────────────────────────────────


def test_health_needs_no_database_or_model() -> None:
    """The container's liveness probe must not depend on Postgres being up, or a
    database blip restarts the API that would have recovered on its own."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_streams_tokens_then_sources_then_done() -> None:
    _seed_and_override(CITED)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": f"Did {NAME_EN} have trouble?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds.count("token") > 1  # it is a stream, not one blob
    assert kinds[-2:] == ["sources", "done"]


def test_the_streamed_tokens_join_into_the_answer() -> None:
    _seed_and_override(CITED)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": f"Did {NAME_EN} have trouble?"})

    tokens = "".join(str(data["text"]) for kind, data in _events(response.text) if kind == "token")
    assert tokens == CITED


def test_sources_arrive_with_enough_to_render_a_card() -> None:
    _seed_and_override(CITED)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": f"Did {NAME_EN} have trouble?"})

    sources = next(data for kind, data in _events(response.text) if kind == "sources")
    first = sources["sources"][0]  # type: ignore[index]
    assert first["marker"] == "S1"
    assert TROUBLE in first["text"]
    assert first["reference"].startswith("incident_comment:")


def test_done_reports_the_route_and_whether_it_abstained() -> None:
    """The UI shows the route, and an eval harness scores it — so it is part of the
    contract, not debug output."""
    _seed_and_override(CITED)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": f"Did {NAME_EN} have trouble?"})

    done = next(data for kind, data in _events(response.text) if kind == "done")
    assert done["route"] == "vector"
    assert done["abstained"] is False
    assert done["horse_id"] == HORSE_ID


def test_a_refusal_streams_like_any_other_answer() -> None:
    """The client should not need a second code path to display "I don't know"."""
    _seed_and_override(CITED)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Did NOTAREALHORSE have trouble?"})

    events = _events(response.text)
    tokens = "".join(str(data["text"]) for kind, data in events if kind == "token")
    done = next(data for kind, data in events if kind == "done")

    assert tokens == NO_EVIDENCE
    assert done["abstained"] is True


def test_an_empty_question_is_a_422_not_a_stream() -> None:
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_an_overlong_question_is_rejected() -> None:
    """A prompt-sized question is either an accident or an attempt to run up a bill."""
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "x" * 5000})

    assert response.status_code == 422


def test_an_unconfigured_provider_is_an_event_not_a_crash() -> None:
    """A demo with an expired key should say so on the question, not fail to boot —
    and the 200 has already gone out, so it cannot be a status code."""
    _seed_and_override(CITED)
    app.dependency_overrides[get_llm_dependency] = lambda: None

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": f"Did {NAME_EN} have trouble?"})

    events = _events(response.text)
    assert events == [("error", {"message": "llm_not_configured"})]
