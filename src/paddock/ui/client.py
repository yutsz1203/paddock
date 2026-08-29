"""The demo's side of the HTTP boundary.

The UI is one client of the API among possible many (spec §3), so it holds no
database session and imports nothing from `paddock.db`. Everything it knows about
the corpus arrives over `/coverage` and `/ask`.

## Every failure becomes one sentence

`ApiError` is the only exception that leaves this module. A demo that shows a
traceback because the API was started second, or a bare `422` because a question ran
long, looks broken for a reason that is not its own — and the person reading it is
usually deciding whether the project works.

The refusal the public demo produces most often is 429, from the per-address
limiter. It gets a sentence with the wait in it rather than a status code, because
a visitor who is told to try again in forty seconds will; one shown "429" leaves.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx

from paddock.ui.stream import AnswerStream, parse_sse

ANSWER_TIMEOUT_S = 90.0
"""Generous on purpose. The API verifies the whole answer before sending the first
token (see `paddock.api.routes`), and a rejected first attempt costs a second call —
so time to first token is generation time, twice over in the worst case."""


class ApiError(RuntimeError):
    """The API could not be reached, or refused the request."""


@dataclass(frozen=True)
class CoverageInfo:
    """What `/coverage` answered."""

    meetings: int
    first_date: dt.date | None
    last_date: dt.date | None
    seasons: list[str]


@dataclass(frozen=True)
class ApiClient:
    """A paddock API, over HTTP."""

    http: httpx.Client

    @classmethod
    def at(cls, base_url: str, *, timeout: float = ANSWER_TIMEOUT_S) -> ApiClient:
        return cls(httpx.Client(base_url=base_url, timeout=timeout))

    def coverage(self) -> CoverageInfo:
        """What the corpus holds.

        Raises:
            ApiError: the API is unreachable or answered with an error status.
        """
        try:
            response = self.http.get("/coverage")
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ApiError(self._status_message(error.response)) from error
        except httpx.HTTPError as error:
            raise ApiError(self._unreachable_message()) from error

        payload = response.json()
        return CoverageInfo(
            meetings=payload["meetings"],
            first_date=_as_date(payload["first_date"]),
            last_date=_as_date(payload["last_date"]),
            seasons=list(payload["seasons"]),
        )

    @contextmanager
    def ask(self, question: str) -> Iterator[AnswerStream]:
        """Ask one question, streaming the answer.

        The connection stays open for as long as the context is, because the stream
        is read lazily — iterate it inside the `with`, not after.

        Raises:
            ApiError: the API is unreachable, refused the question, or the
                connection dropped part-way through the answer.
        """
        try:
            with self.http.stream("POST", "/ask", json={"question": question}) as response:
                if response.status_code != httpx.codes.OK:
                    response.read()
                    raise ApiError(self._status_message(response))
                yield AnswerStream(parse_sse(response.iter_lines()))
        except httpx.HTTPError as error:
            raise ApiError(self._unreachable_message()) from error

    def _unreachable_message(self) -> str:
        return f"The API at {self.http.base_url} did not answer. Is it running?"

    def _status_message(self, response: httpx.Response) -> str:
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            return _rate_limited_message(response.headers.get("Retry-After"))
        return f"The API at {self.http.base_url} refused the request ({response.status_code})."


def _rate_limited_message(retry_after: str | None) -> str:
    """The one refusal a visitor is likely to meet, so it gets its own sentence.

    The demo is rate limited per address, and "refused the request (429)" names a
    status code the reader cannot act on. `Retry-After` is whole seconds when the
    API sends it, and the API always does — but a proxy in between can drop a
    header, so the message still has to make sense without it.
    """
    wait = f" Try again in {retry_after} seconds." if retry_after else " Try again shortly."
    return "Too many questions from this address." + wait


def _as_date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None
