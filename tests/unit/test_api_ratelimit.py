"""`/ask` under a burst.

No database and no model. A refused request never reaches the handler, which is the
point of putting the limiter in a dependency rather than inside `_stream`. An
*allowed* request would reach it, so the provider is overridden to None — `_stream`
then reports `llm_not_configured` and returns before it opens a session. That keeps
this file in the unit suite, which CI runs without Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from paddock.api.main import app
from paddock.api.ratelimit import RateLimiter
from paddock.api.routes import get_llm_dependency

QUESTION = {"question": "Did GROUPER have any trouble last time?"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    # State lives on the app object, which is module-level and outlives one test.
    for leftover in ("rate_limiter", "trusted_proxy_hops"):
        if hasattr(app.state, leftover):
            delattr(app.state, leftover)
    app.dependency_overrides[get_llm_dependency] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_the_limiter_is_built_at_startup_from_settings(client: TestClient) -> None:
    """Nothing configures it per request, so a missing wire-up fails here."""
    assert isinstance(app.state.rate_limiter, RateLimiter)
    assert app.state.rate_limiter.limit > 0


def _limit_to(client: TestClient, calls: int, *, hops: int = 0) -> None:
    app.state.rate_limiter = RateLimiter(limit=calls, window_s=60.0)
    app.state.trusted_proxy_hops = hops


def test_a_burst_past_the_limit_is_refused(client: TestClient) -> None:
    _limit_to(client, 0)

    response = client.post("/ask", json=QUESTION)

    assert response.status_code == 429


def test_a_refusal_says_when_to_come_back(client: TestClient) -> None:
    _limit_to(client, 0)

    response = client.post("/ask", json=QUESTION)

    assert int(response.headers["retry-after"]) >= 1


def test_a_refusal_reads_as_a_sentence_not_a_stack_trace(client: TestClient) -> None:
    _limit_to(client, 0)

    response = client.post("/ask", json=QUESTION)

    assert "too many" in response.json()["detail"].lower()


def test_health_is_not_limited(client: TestClient) -> None:
    """A liveness probe that trips the limiter restarts a healthy container."""
    _limit_to(client, 0)

    assert client.get("/health").status_code == 200


def test_a_forwarded_address_is_ignored_when_no_proxy_is_trusted(client: TestClient) -> None:
    """Otherwise one header per request gives every caller a fresh budget."""
    _limit_to(client, 1, hops=0)
    client.post("/ask", json=QUESTION, headers={"X-Forwarded-For": "9.9.9.1"})

    response = client.post("/ask", json=QUESTION, headers={"X-Forwarded-For": "9.9.9.2"})

    assert response.status_code == 429


def test_a_forwarded_address_is_metered_when_a_proxy_is_trusted(client: TestClient) -> None:
    _limit_to(client, 1, hops=1)
    first = client.post("/ask", json=QUESTION, headers={"X-Forwarded-For": "9.9.9.1"})

    second = client.post("/ask", json=QUESTION, headers={"X-Forwarded-For": "9.9.9.2"})

    assert first.status_code != 429
    assert second.status_code != 429


def test_a_spoofed_address_cannot_buy_a_second_budget(client: TestClient) -> None:
    """One trusted hop means Caddy appended the real address after the caller's."""
    _limit_to(client, 1, hops=1)
    client.post("/ask", json=QUESTION, headers={"X-Forwarded-For": "1.2.3.4, 9.9.9.1"})

    response = client.post("/ask", json=QUESTION, headers={"X-Forwarded-For": "5.6.7.8, 9.9.9.1"})

    assert response.status_code == 429
