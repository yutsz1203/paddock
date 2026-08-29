"""The daily cap on model calls, and the wrapper that spends it.

## What this guards, and what it does not

Spec §477 asks the public demo for a hard spend cap. There are two caps and they
sit in different places, so it is worth being exact about which one this is.

**The cap that cannot be evaded is at the provider.** A billing limit on the Gemini
key stops spending whatever this process does, including if it is restarted in a
loop. Setting it is a step in the deploy runbook, not a line of Python, and it is
the one that makes "hard" true.

**This is the application half.** It exists so that an exhausted budget reads as a
sentence rather than a 500 from an SDK three seconds into someone's question, and
so the demo stops short of the provider limit rather than discovering it. Spec
§477's "graceful degradation, not a stack trace" is this module's job.

## In-process, and what that costs

The count lives in memory, like the rate limiter beside it, so it resets when the
container restarts. A crash loop with traffic on it would therefore spend more than
the cap. That hole is closed by the provider-side limit above, not here.

The tempting fix — a row in Postgres — was rejected for a specific reason. The
committed demo dump (`data/seed/`) carries its own schema and is restored without
migrations, so a new table would make `/ask` fail against a fresh clone until the
22 MB dump was rebuilt. A guard that breaks `make demo` is a bad trade for closing
a hole the provider already closes.

## Counted in calls, not tokens

One question is one or two model calls: the graph retries synthesis once when the
citation check rejects the first answer. Tokens would be the truer unit of spend,
and reading them back means a non-streaming response or a provider-specific usage
field — neither of which the `LLM` protocol has, and adding one would put billing
into an interface that deliberately knows nothing but `stream`. Calls are a
coarse-but-honest proxy: answers here are short and bounded by `MAX_TOKENS`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator, Sequence

from paddock.llm.provider import LLM, Message


class DailyCapReachedError(RuntimeError):
    """The API has made as many model calls today as it is allowed."""


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


class DailyCallBudget:
    """How many model calls the API may make in one UTC day."""

    def __init__(self, *, cap: int, today: Callable[[], dt.date] = _utc_today) -> None:
        """
        Args:
            cap: calls allowed per day. Zero refuses everything. Negative means no
                cap — the escape hatch for a developer running against their own
                key, and never what the deployed box is configured with.
            today: the current date. Injected so the reset is testable without
                waiting for midnight.

        The day is UTC rather than Hong Kong time, deliberately. Providers meter and
        reset on UTC days, so a cap on a local day would drift out of step with the
        limit it is meant to stay under.
        """
        self.cap = cap
        self._today = today
        self._day = today()
        self._calls = 0

    @property
    def calls_today(self) -> int:
        self._roll_over()
        return self._calls

    @property
    def exhausted(self) -> bool:
        """Whether the next call would be refused.

        Reading this spends nothing, which is what lets `/ask` ask before it opens a
        database session or loads the query encoder.
        """
        if self.cap < 0:
            return False
        self._roll_over()
        return self._calls >= self.cap

    def spend(self) -> None:
        """Charge one model call.

        Raises:
            DailyCapReachedError: the cap for today is already spent.
        """
        if self.exhausted:
            raise DailyCapReachedError(f"the daily cap of {self.cap} model calls is spent")
        self._calls += 1

    def _roll_over(self) -> None:
        now = self._today()
        if now != self._day:
            self._day = now
            self._calls = 0


class CappedLLM:
    """An `LLM` that charges the budget before it calls the provider.

    A wrapper rather than a check inside the agent, because the agent should not
    know that its model costs money — and because the wrapper cannot be forgotten at
    one of the two call sites the retry loop creates.
    """

    def __init__(self, inner: LLM, budget: DailyCallBudget) -> None:
        # Copied rather than a property: `LLM` declares `name` as an attribute, and
        # a read-only property does not satisfy that, so mypy stops seeing this
        # class as an `LLM`. Logs and traces should name the provider, not the
        # wrapper around it.
        self.name = inner.name
        self._inner = inner
        self._budget = budget

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield the reply, having first charged one call.

        Raises:
            DailyCapReachedError: on the first `next()`, before the provider is touched.

        The charge happens on iteration and not when this method is called, because
        `stream` returns a generator: charging at call time would bill for a stream
        that a caller built and then abandoned.
        """
        self._budget.spend()
        yield from self._inner.stream(messages)
