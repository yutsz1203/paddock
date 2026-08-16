"""The answer path: resolve → route → retrieve → synthesise → enforce.

This is the smallest graph that can be honest. It exists to prove the architecture
works end to end, not to answer well — the router is keywords, the retrieval is the
minimal pair from T8, and both are replaced in Phase 4 (T15 to T17). What is *not*
provisional is the enforcement: no answer reaches a reader without a citation, and
that rule is the same one the finished system will have.

## Why a graph rather than five function calls

Four of the five steps are a straight line, and a straight line does not need
LangGraph. The fifth is not: enforcement either finishes, or sends the state back to
synthesis with a correction, once. That edge is the control flow, and having it
declared — rather than living in a `while attempts < 2` somewhere — is what makes the
path traceable step by step in Langfuse (T18) and arguable in an interview.

## The order of the two guards

Retrieval happens before the model is called, and if it returns nothing the graph
abstains without calling the model at all. That is not only cheaper: an LLM handed
an empty evidence block and a question about a horse will answer it, and the answer
will be fluent, plausible and invented. Removing the call removes the path.

## Regenerate once, then stop

A first answer that cites nothing is usually a formatting slip, and one correction
naming the offending sentence fixes it. A second failure means the model is
reasoning from memory, and the honest output is the abstention. There is no third
attempt: each one costs a call and a second of latency, and the marginal chance of
a grounded answer is not worth either.

## Scope: a question is about one horse

If no known horse name appears in the question, the graph abstains. That is a real
limit — spec §1 Q6 ("best jockey at Happy Valley") has no horse in it and will be
refused until T15 adds `jockey_stats` and T16 widens the router. Refusing is the
right failure while that is true: answering a jockey question from one horse's
comments would be worse than saying no.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from paddock.agent.citations import (
    NO_EVIDENCE,
    Source,
    SourceKind,
    Verdict,
    number_sources,
    verify,
)
from paddock.agent.prompts import SYSTEM, correction_prompt, question_prompt
from paddock.embed.embedder import Embedder
from paddock.llm.provider import LLM, Message, complete
from paddock.retrieval.sql_tool import RunLine, find_horse, recent_runs
from paddock.retrieval.vector_tool import CommentHit, search_comments

Route = Literal["sql", "vector", "both"]

MAX_ATTEMPTS = 2
"""One answer, one correction. See the module docstring."""

RUNS_PER_ANSWER = 5
COMMENTS_PER_ANSWER = 5

MAX_DISTANCE = 0.6
"""Cosine distance beyond which a comment is not evidence about this question.

A provisional number. T20 tunes it against the golden set; until then it is set
where bge-m3 puts loosely-related racing prose, which is conservative — it prefers
abstaining to answering from a comment that merely sounds relevant.
"""

# Words that mean "what happened during the race", which only the narrative knows.
_TROUBLE_WORDS = (
    "trouble",
    "interfer",
    "hamper",
    "incident",
    "excuse",
    "steward",
    "checked",
    "bumped",
    "baulked",
    "crowded",
    "vet",
    "lame",
    "bled",
    "unlucky",
    "wide",
    "受阻",
    "阻礙",
    "騎師報告",
    "獸醫",
    "受傷",
    "沖擊",
    "藉口",
)

# Words that mean "count, filter and sort", which only SQL does correctly.
_FORM_WORDS = (
    "record",
    "statistic",
    "strike rate",
    "win rate",
    "how many",
    "average",
    "往績",
    "勝出次數",
    "命中率",
)

_VENUES = ("sha tin", "happy valley", "沙田", "跑馬地")


@dataclass(frozen=True)
class Answer:
    """What the API returns, and what a trace records."""

    text: str
    sources: list[Source]
    route: Route
    horse_id: str | None
    horse_name: str | None
    abstained: bool
    attempts: int
    """Model calls made. 0 when the graph abstained without asking."""


class _State(TypedDict, total=False):
    question: str
    horse_id: str | None
    horse_name: str | None
    route: Route
    sources: list[Source]
    answer: str
    verdict: Verdict | None
    attempts: int
    abstained: bool


def answer_question(session: Session, *, question: str, llm: LLM, embedder: Embedder) -> Answer:
    """Answer one question, with citations, or refuse.

    Args:
        session: an open session.
        question: the user's question, English or Chinese.
        llm: the provider to synthesise with.
        embedder: the model that embedded the corpus — the same one, or the vectors
            are not comparable.

    Raises:
        ValueError: the question is empty.
    """
    if not question.strip():
        raise ValueError("question is empty")

    graph = _build(session, llm, embedder)
    final: _State = graph.invoke({"question": question.strip(), "attempts": 0})

    return Answer(
        text=final["answer"],
        sources=final.get("sources", []),
        route=final.get("route", "both"),
        horse_id=final.get("horse_id"),
        horse_name=final.get("horse_name"),
        abstained=final.get("abstained", False),
        attempts=final.get("attempts", 0),
    )


def _build(session: Session, llm: LLM, embedder: Embedder) -> Any:
    """Compile the graph with its dependencies closed over.

    Session, model and embedder are captured here rather than carried in the state:
    the state is then plain data, which is what makes it printable in a trace and
    comparable in a test. Compiling is a handful of dict operations, so doing it per
    request costs nothing measurable and keeps request scope honest.
    """
    builder = StateGraph(_State)

    builder.add_node("resolve_entity", lambda state: _resolve_entity(session, state))
    builder.add_node("retrieve", lambda state: _retrieve(session, embedder, state))
    builder.add_node("route", _route)
    builder.add_node("synthesise", lambda state: _synthesise(llm, state))
    builder.add_node("enforce", _enforce)
    builder.add_node("abstain", _abstain)

    builder.set_entry_point("resolve_entity")
    builder.add_conditional_edges(
        "resolve_entity",
        lambda state: "route" if state.get("horse_id") else "abstain",
        {"route": "route", "abstain": "abstain"},
    )
    builder.add_edge("route", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        lambda state: "synthesise" if state.get("sources") else "abstain",
        {"synthesise": "synthesise", "abstain": "abstain"},
    )
    builder.add_edge("synthesise", "enforce")
    builder.add_conditional_edges(
        "enforce",
        _after_enforce,
        {
            "synthesise": "synthesise",
            "abstain": "abstain",
            "done": END,
        },
    )
    builder.add_edge("abstain", END)

    return builder.compile()


# ── Nodes ───────────────────────────────────────────────────────────────────────


def _resolve_entity(session: Session, state: _State) -> _State:
    match = find_horse(session, state["question"])
    if match is None:
        return {"horse_id": None, "horse_name": None}
    return {"horse_id": match.horse_id, "horse_name": match.name_en or match.name_zh}


def _route(state: _State) -> _State:
    """Keywords, for now.

    Crude and honest about it: T16 replaces this with a router evaluated against the
    golden set. The shape it has to get right today is the one the spec argues for —
    a question about what happened in running must not be answered from a results
    table, and a question about the last five runs at a distance must not be answered
    from prose.
    """
    question = state["question"].lower()

    trouble = any(word in question for word in _TROUBLE_WORDS)
    structured = (
        any(word in question for word in _FORM_WORDS)
        or any(venue in question for venue in _VENUES)
        or _mentions_a_distance(question)
        or _mentions_a_count(question)
    )

    if trouble and not structured:
        return {"route": "vector"}
    if structured and not trouble:
        return {"route": "sql"}
    return {"route": "both"}


def _retrieve(session: Session, embedder: Embedder, state: _State) -> _State:
    horse_id = state["horse_id"]
    assert horse_id is not None  # the conditional edge guarantees it
    route = state["route"]

    runs: list[RunLine] = []
    if route in ("sql", "both"):
        runs = recent_runs(session, horse_id=horse_id, n=RUNS_PER_ANSWER)

    hits: list[CommentHit] = []
    if route in ("vector", "both"):
        hits = [
            hit
            for hit in search_comments(
                session,
                query=state["question"],
                embedder=embedder,
                horse_id=horse_id,
                limit=COMMENTS_PER_ANSWER,
            )
            if hit.distance <= MAX_DISTANCE
        ]

    items: list[tuple[SourceKind, str, str]] = [
        ("run", _describe_run(run), f"race:{run.race_id}") for run in runs
    ]
    items += [("comment", _describe_hit(hit), f"incident_comment:{hit.comment_id}") for hit in hits]

    return {"sources": number_sources(items)}


def _synthesise(llm: LLM, state: _State) -> _State:
    messages = [
        Message(role="system", content=SYSTEM),
        Message(role="user", content=question_prompt(state["question"], state["sources"])),
    ]

    verdict = state.get("verdict")
    if verdict is not None:
        # The rejected answer and the reason, so the correction has something to fix.
        messages.append(Message(role="assistant", content=state["answer"]))
        messages.append(Message(role="user", content=correction_prompt(verdict)))

    return {"answer": complete(llm, messages).strip(), "attempts": state["attempts"] + 1}


def _enforce(state: _State) -> _State:
    return {"verdict": verify(state["answer"], state["sources"])}


def _after_enforce(state: _State) -> str:
    verdict = state["verdict"]
    assert verdict is not None
    if verdict.grounded:
        return "done"
    return "synthesise" if state["attempts"] < MAX_ATTEMPTS else "abstain"


def _abstain(state: _State) -> _State:
    return {"answer": NO_EVIDENCE, "abstained": True, "sources": []}


# ── Formatting evidence ─────────────────────────────────────────────────────────


def _describe_run(run: RunLine) -> str:
    """A form line as one sentence, because the model quotes it as one.

    Everything a reader needs to check the claim is here — the date and venue locate
    the race, the placing carries its field size — so a source card is renderable
    from the citation alone.
    """
    parts = [
        run.race_date.isoformat(),
        run.racecourse,
        f"R{run.race_no}",
        f"{run.distance_m}m" if run.distance_m else None,
        run.race_class,
        f"going {run.going}" if run.going else None,
        f"finished {run.finish_pos} of {run.field_size}" if run.finish_pos else "did not finish",
        f"beaten {run.margin}L" if run.margin else None,
        f"draw {run.draw}" if run.draw else None,
        f"carried {run.carried_weight_lb}lb" if run.carried_weight_lb else None,
        f"jockey {run.jockey}" if run.jockey else None,
        f"SP {run.win_odds}" if run.win_odds else None,
    ]
    return ", ".join(part for part in parts if part)


def _describe_hit(hit: CommentHit) -> str:
    """The stewards' words, with just enough context to place them."""
    where = f"{hit.race_date.isoformat()} {hit.racecourse} R{hit.race_no}"
    return f"Stewards' report, {where}: {hit.text}"


def _mentions_a_distance(question: str) -> bool:
    return re.search(r"\b\d{3,4}\s*(m|metre|meter|米|米賽)\b", question) is not None


def _mentions_a_count(question: str) -> bool:
    return re.search(r"\b(last|past|最近)\s*\d+\b", question) is not None
