"""`/ask` with the daily cap spent.

No database. An exhausted budget is reported before `_stream` opens a session, which
is the behaviour under test as much as the message is: a demo that has run out of
budget should not also be loading a 2.2 GB encoder for every visitor.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from tests.doubles import ScriptedLLM

from paddock.api.budget import DailyCallBudget
from paddock.api.main import app
from paddock.api.routes import get_llm_dependency

QUESTION = {"question": "Did GROUPER have any trouble last time?"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    for leftover in ("budget", "rate_limiter", "trusted_proxy_hops"):
        if hasattr(app.state, leftover):
            delattr(app.state, leftover)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _spent(client: TestClient) -> None:
    app.state.budget = DailyCallBudget(cap=0)
    app.dependency_overrides[get_llm_dependency] = lambda: ScriptedLLM("never used")


def test_a_spent_cap_is_an_event_and_not_a_crash(client: TestClient) -> None:
    _spent(client)

    response = client.post("/ask", json=QUESTION)

    assert response.status_code == 200
    assert "daily_cap_reached" in response.text


def test_a_spent_cap_never_reaches_the_provider(client: TestClient) -> None:
    _spent(client)
    llm = ScriptedLLM("never used")
    app.dependency_overrides[get_llm_dependency] = lambda: llm

    client.post("/ask", json=QUESTION)

    assert llm.calls == []


def test_the_budget_is_built_at_startup_from_settings(client: TestClient) -> None:
    assert isinstance(app.state.budget, DailyCallBudget)


def test_a_spent_cap_still_reports_the_missing_provider_first(client: TestClient) -> None:
    """Two problems at once. The one the operator can fix is the one to name."""
    app.state.budget = DailyCallBudget(cap=0)
    app.dependency_overrides[get_llm_dependency] = lambda: None

    response = client.post("/ask", json=QUESTION)

    assert "llm_not_configured" in response.text
