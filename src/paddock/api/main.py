"""The ASGI app.

Thin by design: the API is a transport over `agent.graph`, and everything that could
be called business logic lives below it. Run with::

    uv run uvicorn paddock.api.main:app --reload

## Startup builds the provider; it does not require one

The LLM client is constructed once at startup, so a missing key is reported in the
first line of the log rather than in someone's first question. It is a warning and
not a crash: `/health` should answer, and the demo should say "not configured"
rather than refuse to boot — which is also what makes the API testable without a
key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from paddock.api.routes import get_embedder_dependency, get_llm_dependency, router
from paddock.config import get_settings
from paddock.embed.embedder import get_embedder
from paddock.llm.provider import build_llm

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    try:
        app.state.llm = build_llm(settings)
        log.info("llm_ready", provider=settings.llm_provider, model=settings.llm_model)
    except ValueError as error:
        # Not fatal: /health still answers and /ask reports the problem per request.
        log.warning("llm_not_configured", error=str(error))
        app.state.llm = None

    # Cheap — the 2.2 GB model is not read until the first embed call.
    app.state.embedder = get_embedder()
    yield


app = FastAPI(
    title="paddock",
    version="0.1.0",
    summary="Bilingual question answering over HKJC racing data, with citations.",
    lifespan=lifespan,
)
app.include_router(router)

__all__ = ["app", "get_embedder_dependency", "get_llm_dependency"]
