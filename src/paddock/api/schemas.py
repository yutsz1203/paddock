"""Request and response shapes.

Pydantic at the boundary, dataclasses inside (spec §8). The validation here is not
ceremony: `question` is the only thing a stranger controls, and both of its limits
have a reason — an empty question would reach the graph and raise, and a very long
one is either an accident or an attempt to spend someone else's tokens.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 500
"""Longer than any real racing question, short enough that a paste attack is cheap
to refuse. The rate limiter in T24 is the other half of this."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question is empty")
        return stripped


class SourceOut(BaseModel):
    """A citation, as a client renders it."""

    marker: str
    kind: str
    text: str
    reference: str


class CoverageOut(BaseModel):
    """What the demo holds, as its banner states it."""

    meetings: int
    first_date: dt.date | None
    last_date: dt.date | None
    seasons: list[str]
