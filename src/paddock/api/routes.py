"""`/ask`, `/coverage` and `/health`.

## The stream carries a verified answer, not a live one

Tokens are emitted only after the graph has produced an answer that passed the
citation check. Enforcement needs the whole text, and spec §10 admits no uncited
claim — so streaming as the model writes would mean publishing sentences we might
then have to retract. Time to first token therefore includes generation.

That is a knowing trade, not an oversight. The alternative that keeps both
properties is to enforce sentence by sentence and release each one as it passes,
which is worth doing when there is an eval to show it does not change groundedness
(T16). Until then the guarantee wins over the latency.

## Event protocol

``token`` (repeatedly) → ``sources`` (once) → ``done`` (once). A refusal uses the
same three events, so a client needs no second code path to display "I don't know".
``error`` replaces ``done`` if something fails mid-stream, because an HTTP status
cannot: the 200 has already been sent with the first byte.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from paddock.agent.graph import answer_question
from paddock.api.schemas import AskRequest, CoverageOut, SourceOut
from paddock.db.coverage import corpus_coverage
from paddock.db.session import session_scope
from paddock.embed.embedder import Embedder, get_embedder
from paddock.llm.provider import LLM

log = structlog.get_logger(__name__)
router = APIRouter()

# Whitespace-preserving pieces, so the client can concatenate what it receives and
# get the answer back exactly.
_PIECES = re.compile(r"\S+\s*")


def get_llm_dependency(request: Request) -> LLM | None:
    """The provider built at startup, or None if it could not be.

    Deliberately a cheap attribute read that cannot fail. FastAPI resolves
    dependencies alongside body validation, so a dependency that constructs a client
    — or raises because a key is missing — turns a malformed request into a 500
    instead of a 422. Configuration problems belong to startup and to the stream,
    not to input validation.
    """
    return getattr(request.app.state, "llm", None)


def get_embedder_dependency(request: Request) -> Embedder:
    return getattr(request.app.state, "embedder", None) or get_embedder()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness only.

    Deliberately touches neither Postgres nor the embedding model: a readiness probe
    that fails on a database blip restarts an API process that would have recovered
    by itself.
    """
    return {"status": "ok"}


@router.get("/coverage")
def coverage() -> CoverageOut:
    """What the corpus holds, so a client can say so before anyone asks a question.

    Reads the database, unlike `/health`. That is the point: a visitor cannot judge
    a refusal without knowing the range it was refused against, and a range that is
    a constant somewhere in the UI stops being true the week the live pipeline runs.
    """
    with session_scope() as session:
        return CoverageOut(**vars(corpus_coverage(session)))


@router.post("/ask")
def ask(
    body: AskRequest,
    llm: Annotated[LLM | None, Depends(get_llm_dependency)],
    embedder: Annotated[Embedder, Depends(get_embedder_dependency)],
) -> StreamingResponse:
    """Answer a question, streaming the verified answer over SSE."""
    return StreamingResponse(
        _stream(body.question, llm, embedder),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _stream(question: str, llm: LLM | None, embedder: Embedder) -> Iterator[str]:
    if llm is None:
        # Startup already logged why. Reported as an event rather than a status code
        # because the 200 went out with the response headers.
        yield _event("error", {"message": "llm_not_configured"})
        return

    try:
        with session_scope() as session:
            answer = answer_question(session, question=question, llm=llm, embedder=embedder)
    except Exception as error:
        log.exception("ask_failed", question=question)
        yield _event("error", {"message": type(error).__name__})
        return

    log.info(
        "answered",
        route=answer.route,
        horse_id=answer.horse_id,
        abstained=answer.abstained,
        attempts=answer.attempts,
        sources=len(answer.sources),
    )

    for piece in _PIECES.findall(answer.text):
        yield _event("token", {"text": piece})

    yield _event(
        "sources",
        {"sources": [SourceOut(**vars(source)).model_dump() for source in answer.sources]},
    )
    yield _event(
        "done",
        {
            "route": answer.route,
            "horse_id": answer.horse_id,
            "horse_name": answer.horse_name,
            "abstained": answer.abstained,
            "attempts": answer.attempts,
        },
    )


def _event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
