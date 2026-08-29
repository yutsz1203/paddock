"""The daily call cap, and the wrapper that spends it."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence

import pytest

from paddock.api.budget import CappedLLM, DailyCallBudget, DailyCapReachedError
from paddock.llm.provider import Message, complete


class FakeCalendar:
    def __init__(self) -> None:
        self.today = dt.date(2026, 8, 29)

    def __call__(self) -> dt.date:
        return self.today

    def tomorrow(self) -> None:
        self.today += dt.timedelta(days=1)


class CountingLLM:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        self.calls += 1
        yield "an "
        yield "answer"


def _budget(cap: int = 2) -> tuple[DailyCallBudget, FakeCalendar]:
    calendar = FakeCalendar()
    return DailyCallBudget(cap=cap, today=calendar), calendar


def test_calls_up_to_the_cap_are_spent() -> None:
    budget, _ = _budget(cap=2)

    budget.spend()
    budget.spend()

    assert budget.calls_today == 2


def test_the_call_past_the_cap_raises() -> None:
    budget, _ = _budget(cap=1)
    budget.spend()

    with pytest.raises(DailyCapReachedError):
        budget.spend()


def test_the_count_resets_on_the_next_day() -> None:
    budget, calendar = _budget(cap=1)
    budget.spend()

    calendar.tomorrow()

    assert budget.exhausted is False
    budget.spend()
    assert budget.calls_today == 1


def test_exhausted_is_readable_without_spending() -> None:
    """`_stream` asks before it opens a session, so asking must cost nothing."""
    budget, _ = _budget(cap=1)

    assert budget.exhausted is False
    assert budget.exhausted is False

    budget.spend()
    assert budget.exhausted is True


def test_a_cap_of_zero_is_exhausted_from_the_start() -> None:
    budget, _ = _budget(cap=0)

    assert budget.exhausted is True


def test_a_negative_cap_means_no_cap() -> None:
    """The escape hatch for local development, where the key is the developer's own."""
    budget, _ = _budget(cap=-1)

    for _ in range(1000):
        budget.spend()

    assert budget.exhausted is False


def test_the_wrapper_spends_one_call_per_stream() -> None:
    budget, _ = _budget(cap=5)
    inner = CountingLLM()

    complete(CappedLLM(inner, budget), [Message(role="user", content="hello")])

    assert budget.calls_today == 1
    assert inner.calls == 1


def test_the_wrapper_returns_what_the_provider_said() -> None:
    budget, _ = _budget(cap=5)

    text = complete(CappedLLM(CountingLLM(), budget), [Message(role="user", content="hello")])

    assert text == "an answer"


def test_the_wrapper_spends_on_iteration_not_on_construction() -> None:
    """`stream` returns a generator. Charging at call time would bill for a stream
    that a caller built and then abandoned."""
    budget, _ = _budget(cap=5)
    capped = CappedLLM(CountingLLM(), budget)

    stream = capped.stream([Message(role="user", content="hello")])

    assert budget.calls_today == 0
    next(stream)
    assert budget.calls_today == 1


def test_an_exhausted_budget_never_reaches_the_provider() -> None:
    budget, _ = _budget(cap=0)
    inner = CountingLLM()

    with pytest.raises(DailyCapReachedError):
        complete(CappedLLM(inner, budget), [Message(role="user", content="hello")])

    assert inner.calls == 0


def test_the_wrapper_keeps_the_provider_name() -> None:
    """Logs and traces name the provider, not the wrapper around it."""
    budget, _ = _budget(cap=5)

    assert CappedLLM(CountingLLM(), budget).name == "counting"
