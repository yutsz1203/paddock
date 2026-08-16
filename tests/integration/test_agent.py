"""The answer path, end to end, with a scripted model.

Nothing here depends on what an LLM says today. The graph's job is to resolve the
horse, choose a route, retrieve, and then refuse to publish anything the citation
check rejects — and every one of those is deterministic given a fixed reply. What a
real model actually writes is an eval question (T19), not a test question.

The two behaviours that matter most are the ones a demo will be judged on:

**Regenerate once, then abstain.** An answer that cites nothing is not published. The
model gets one corrective attempt, because a missing marker is usually a formatting
slip; a second failure means it is reasoning from memory, and the honest output is
"I don't have evidence for that".

**No evidence, no answer, no call.** With nothing retrieved above threshold the graph
abstains without asking the model anything. Cheaper, and it removes the only path by
which a confident invented answer could reach a reader.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator, Sequence

import pytest
from sqlalchemy import select
from tests.doubles import ScriptedLLM

from paddock.agent.citations import NO_EVIDENCE
from paddock.agent.graph import answer_question
from paddock.db.models import (
    EMBEDDING_DIM,
    Chunk,
    Horse,
    HorseAlias,
    IncidentComment,
    Meeting,
    Race,
    Runner,
)
from paddock.db.session import session_scope
from paddock.embed.store import embed_meeting

pytestmark = pytest.mark.integration

RACE_DATE = dt.date(2098, 11, 2)
HORSE_ID = "HK_2099_Z001"
NAME_EN = "TESTBRED FLYER"
NAME_ZH = "測試飛駒"
FORMER_NAME = "TESTBRED SLOWCOACH"

TROUBLE = "Was hampered approaching the 800 Metres and lost ground."
CLEAN = "Raced wide throughout without cover."

CITED = "It was hampered near the 800 and finished sixth [S1]."
UNCITED = "It looked unlucky and should win next time."


def _axis_vector(similarity: float) -> list[float]:
    vector = [similarity, math.sqrt(max(0.0, 1.0 - similarity**2))]
    return vector + [0.0] * (EMBEDDING_DIM - 2)


class StubEmbedder:
    """Comments sit where they are placed; anything else — the question — sits at the
    near end of the axis, so a placement of 0.95 means "answers this question"."""

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
        session.query(HorseAlias).filter(HorseAlias.horse_id.like("HK_2099_%")).delete(
            synchronize_session=False
        )
        session.query(Horse).filter(Horse.horse_id.like("HK_2099_%")).delete(
            synchronize_session=False
        )


def _seed() -> StubEmbedder:
    """One horse, one run, one comment about trouble and one about nothing much."""
    with session_scope() as session:
        session.add(Horse(horse_id=HORSE_ID, brand_no="Z001", name_en=NAME_EN, name_zh=NAME_ZH))
        session.add(HorseAlias(horse_id=HORSE_ID, name=FORMER_NAME, lang="en"))
        rival = "HK_2099_Z002"
        session.add(Horse(horse_id=rival, brand_no="Z002", name_en="TESTBRED RIVAL"))

        meeting = Meeting(race_date=RACE_DATE, racecourse="ST", going="GOOD")
        session.add(meeting)
        session.flush()
        race = Race(
            meeting_id=meeting.id,
            race_no=5,
            race_class="Class 4",
            distance_m=1200,
            course="A",
            going="GOOD",
            status="finished",
        )
        session.add(race)
        session.flush()

        session.add(
            Runner(
                race_id=race.id,
                horse_id=HORSE_ID,
                horse_no=1,
                draw=3,
                carried_weight_lb=126,
                finish_pos=6,
                margin=4.25,
                win_odds=7.5,
            )
        )
        session.add(Runner(race_id=race.id, horse_id=rival, horse_no=2, finish_pos=1))
        session.add(
            IncidentComment(race_id=race.id, horse_id=HORSE_ID, finish_pos=6, text_en=TROUBLE)
        )
        session.add(IncidentComment(race_id=race.id, horse_id=rival, finish_pos=1, text_en=CLEAN))
        meeting_id = meeting.id

    embedder = StubEmbedder({TROUBLE: 0.95, CLEAN: 0.0})
    with session_scope() as session:
        embed_meeting(session, meeting_id=meeting_id, embedder=embedder)
    return embedder


# ── Entity resolution ───────────────────────────────────────────────────────────


def test_an_english_name_resolves_to_the_horse_id() -> None:
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running last time?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.horse_id == HORSE_ID


def test_a_chinese_name_resolves_to_the_same_horse() -> None:
    """Same horse, same id — the join is the id, never the string (T6)."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"{NAME_ZH} 上仗有冇受阻?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.horse_id == HORSE_ID


def test_a_former_name_still_resolves() -> None:
    """HK horses are renamed mid-career. A question asked with last season's name
    must not come back "no such horse"."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {FORMER_NAME} have any trouble in running?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.horse_id == HORSE_ID


# ── Routing ─────────────────────────────────────────────────────────────────────


def test_a_trouble_question_goes_to_the_comments_only() -> None:
    """Spec §1 Q3. Nothing in a results table answers "did it have an excuse"."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running last time?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.route == "vector"
    assert {source.kind for source in answer.sources} == {"comment"}


def test_a_form_question_goes_to_sql_only() -> None:
    """Spec §1 Q2. "Last 5 runs over 1200m" is ORDER BY and LIMIT — an embedding
    will answer it fluently and wrongly."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"How has {NAME_EN} gone over 1200m at Sha Tin in its last 5 runs?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.route == "sql"
    assert {source.kind for source in answer.sources} == {"run"}


def test_an_open_question_uses_both() -> None:
    """Spec §1 Q1. "How did it perform" wants the result and the reason."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"How did {NAME_EN} perform in its last start?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.route == "both"
    assert {source.kind for source in answer.sources} == {"run", "comment"}


# ── Citations ───────────────────────────────────────────────────────────────────


def test_a_cited_answer_is_published_as_written() -> None:
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.text == CITED
    assert not answer.abstained
    assert answer.attempts == 1


def test_an_uncited_answer_is_regenerated_once() -> None:
    embedder = _seed()
    llm = ScriptedLLM(UNCITED, CITED)

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running?",
            llm=llm,
            embedder=embedder,
        )

    assert answer.text == CITED
    assert answer.attempts == 2
    assert not answer.abstained


def test_the_retry_prompt_names_what_was_wrong() -> None:
    """A bare "try again" gets the same answer back. The correction has to quote the
    sentence that failed."""
    embedder = _seed()
    llm = ScriptedLLM(UNCITED, CITED)

    with session_scope() as session:
        answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running?",
            llm=llm,
            embedder=embedder,
        )

    assert any(UNCITED in prompt for prompt in llm.prompts)


def test_two_uncited_answers_become_an_abstention() -> None:
    """Rather than a third attempt, or publishing the least-bad one."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running?",
            llm=ScriptedLLM(UNCITED, UNCITED),
            embedder=embedder,
        )

    assert answer.text == NO_EVIDENCE
    assert answer.abstained
    assert answer.attempts == 2


def test_an_invented_citation_is_treated_as_ungrounded() -> None:
    embedder = _seed()
    invented = "It won by three lengths [S7]."

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running?",
            llm=ScriptedLLM(invented, invented),
            embedder=embedder,
        )

    assert answer.abstained


def test_sources_resolve_to_rows_that_exist() -> None:
    """A citation that cannot be resolved is decoration. The UI renders these
    directly, so the reference has to name a real row."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question=f"Did {NAME_EN} have any trouble in running?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )
        comment_ids = [
            int(source.reference.split(":")[1])
            for source in answer.sources
            if source.kind == "comment"
        ]
        found = session.scalars(
            select(IncidentComment).where(IncidentComment.id.in_(comment_ids))
        ).all()

    assert comment_ids
    assert len(found) == len(comment_ids)
    assert {row.text_en for row in found} == {TROUBLE}


# ── Abstention ──────────────────────────────────────────────────────────────────


def test_an_unknown_horse_gets_a_refusal_not_a_guess() -> None:
    """The automated half of the plan's T9 verification."""
    embedder = _seed()

    with session_scope() as session:
        answer = answer_question(
            session,
            question="Did NOTAREALHORSE have any trouble in running last time?",
            llm=ScriptedLLM(CITED),
            embedder=embedder,
        )

    assert answer.text == NO_EVIDENCE
    assert answer.abstained
    assert answer.sources == []


def test_abstaining_costs_no_model_call() -> None:
    """With nothing retrieved there is nothing to ground an answer in, so asking the
    model can only produce something invented — and it would be billed."""
    embedder = _seed()
    llm = ScriptedLLM(CITED)

    with session_scope() as session:
        answer_question(
            session,
            question="Did NOTAREALHORSE have any trouble in running last time?",
            llm=llm,
            embedder=embedder,
        )

    assert llm.calls == []


def test_an_empty_question_is_refused() -> None:
    embedder = _seed()

    with session_scope() as session, pytest.raises(ValueError, match="question"):
        answer_question(session, question="   ", llm=ScriptedLLM(CITED), embedder=embedder)
