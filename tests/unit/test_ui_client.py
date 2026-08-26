"""The demo's side of the HTTP boundary.

The UI is one client of the API among possible many (spec §3), so it holds no
database session and imports nothing from `paddock.db`. These tests run against
`httpx.MockTransport`: no server, no network, no Postgres.

What is worth testing here is the failure shapes. A demo that shows a traceback when
the API is down, or a bare 422 when a question is too long, is a demo that looks
broken for reasons that are not its fault.
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from paddock.ui.client import ApiClient, ApiError

COVERAGE = {
    "meetings": 176,
    "first_date": "2024-09-08",
    "last_date": "2026-07-15",
    "seasons": ["2024-25", "2025-26"],
}

ANSWER = (
    'event: token\ndata: {"text": "It was hampered "}\n\n'
    'event: token\ndata: {"text": "near the 800 [S1]."}\n\n'
    'event: sources\ndata: {"sources": [{"marker": "S1", "kind": "comment",'
    ' "text": "Was hampered.", "reference": "incident_comment:77"}]}\n\n'
    'event: done\ndata: {"route": "vector", "horse_id": "HK_2024_K570",'
    ' "horse_name": "SETANTA", "abstained": false, "attempts": 1}\n\n'
)


def _client(handler: object) -> ApiClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return ApiClient(httpx.Client(base_url="http://api.test", transport=transport))


# ── /coverage ───────────────────────────────────────────────────────────────────


def test_coverage_comes_back_as_dates_not_strings() -> None:
    api = _client(lambda request: httpx.Response(200, json=COVERAGE))

    coverage = api.coverage()

    assert coverage.meetings == 176
    assert coverage.first_date == dt.date(2024, 9, 8)
    assert coverage.last_date == dt.date(2026, 7, 15)
    assert coverage.seasons == ["2024-25", "2025-26"]


def test_an_empty_corpus_comes_back_with_no_dates() -> None:
    api = _client(
        lambda request: httpx.Response(
            200, json={"meetings": 0, "first_date": None, "last_date": None, "seasons": []}
        )
    )

    coverage = api.coverage()

    assert coverage.first_date is None
    assert coverage.last_date is None


def test_an_api_that_is_not_running_is_a_readable_error() -> None:
    """The first thing anyone meets on a fresh clone is a UI started before the API.
    A traceback there reads as a broken demo."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    api = _client(refuse)

    with pytest.raises(ApiError) as raised:
        api.coverage()

    assert "http://api.test" in str(raised.value)


# ── /ask ────────────────────────────────────────────────────────────────────────


def test_asking_streams_the_answer_and_keeps_its_sources() -> None:
    api = _client(lambda request: httpx.Response(200, content=ANSWER.encode()))

    with api.ask("Did SETANTA have trouble?") as stream:
        text = "".join(stream)

    assert text == "It was hampered near the 800 [S1]."
    assert [source.marker for source in stream.sources] == ["S1"]
    assert stream.route == "vector"


def test_the_question_is_sent_as_the_body_the_api_validates() -> None:
    seen: dict[str, object] = {}

    def record(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, content=ANSWER.encode())

    api = _client(record)
    with api.ask("Did SETANTA have trouble?") as stream:
        list(stream)

    assert seen["url"] == "http://api.test/ask"
    assert seen["body"] == {"question": "Did SETANTA have trouble?"}


def test_a_rejected_question_is_a_readable_error_not_a_traceback() -> None:
    """422 is the API refusing an empty or overlong question. The visitor typed it,
    so they get a sentence rather than a status code."""
    api = _client(lambda request: httpx.Response(422, json={"detail": "too long"}))

    with pytest.raises(ApiError) as raised:  # noqa: SIM117
        with api.ask("x" * 5000) as stream:
            list(stream)

    assert "422" in str(raised.value)


def test_a_stream_that_dies_mid_answer_is_a_readable_error() -> None:
    def cut(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    api = _client(cut)

    with pytest.raises(ApiError), api.ask("Did SETANTA have trouble?") as stream:
        list(stream)
