"""The politeness guarantees: one request a second, and retries that back off.

These live in the client rather than in each caller because a caller that forgets is
indistinguishable from an attack. The backfill makes ~3,700 requests, so the rules
here are the difference between a slow afternoon and being blocked.

What is worth retrying is narrower than "anything that raised". A 503 is HKJC having
a moment and a second attempt will probably work. A 404 will still be a 404 in eight
seconds, and during backfill — where candidate dates are generated and most are not
meetings — retrying those would triple the cost of every miss.

No network: `httpx.MockTransport` answers every request, so these run in CI.
"""

from __future__ import annotations

import time

import httpx
import pytest

from paddock.config import Settings
from paddock.ingest.http import HkjcClient

# Fast enough to keep the suite quick, slow enough that the assertions mean something.
FAST = Settings(hkjc_request_delay_s=0.05, hkjc_retry_backoff_s=0.01, hkjc_max_retries=3)


def _client(handler: object, settings: Settings = FAST) -> HkjcClient:
    return HkjcClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_requests_are_spaced_by_the_politeness_delay() -> None:
    client = _client(lambda request: httpx.Response(200, text="ok"))

    started = time.monotonic()
    with client:
        client.get_text("/one")
        client.get_text("/two")
    elapsed = time.monotonic() - started

    assert elapsed >= FAST.hkjc_request_delay_s, "the second request must wait its turn"


def test_the_first_request_does_not_wait() -> None:
    """The delay is between requests, not before them — 88 meetings pay it 87 times."""
    client = _client(lambda request: httpx.Response(200, text="ok"))

    started = time.monotonic()
    with client:
        client.get_text("/one")

    assert time.monotonic() - started < FAST.hkjc_request_delay_s


def test_a_transport_error_is_retried_and_can_succeed() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, text="second time lucky")

    with _client(handler) as client:
        assert client.get_text("/flaky") == "second time lucky"
    assert attempts == 2


def test_a_server_error_is_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts == 1 else 200, text="ok")

    with _client(handler) as client:
        assert client.get_text("/flaky") == "ok"
    assert attempts == 2


def test_a_missing_page_is_not_retried() -> None:
    """A 404 will still be a 404 in eight seconds. Backfill generates many of them."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="gone")

    with _client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        client.get_text("/missing")
    assert attempts == 1, "retrying a 404 spends the politeness budget on nothing"


def test_it_gives_up_after_the_configured_number_of_attempts() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="down")

    with _client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        client.get_text("/down")
    assert attempts == FAST.hkjc_max_retries


def test_the_wait_between_attempts_grows() -> None:
    """Exponential, not fixed: hammering a struggling server at a steady rate is rude."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    settings = Settings(hkjc_request_delay_s=0.0, hkjc_retry_backoff_s=0.05, hkjc_max_retries=3)
    started = time.monotonic()
    with _client(handler, settings) as client, pytest.raises(httpx.HTTPStatusError):
        client.get_text("/down")

    # Two waits between three attempts: 0.05 then 0.10, so a fixed 0.05 would fail this.
    assert time.monotonic() - started >= 0.15


def test_the_url_the_archive_keys_on_is_the_one_that_gets_requested() -> None:
    """A separately assembled URL would split one page's history in two."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, text="ok")

    with _client(handler) as client:
        expected = client.url_for("/report", {"date": "2026/04/26"})
        client.get_text("/report", {"date": "2026/04/26"})

    assert requested == [expected]
