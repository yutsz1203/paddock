"""The per-IP limiter, on its own.

A sliding window rather than a fixed one. A fixed window lets a caller spend the
whole budget in the last second of one window and the whole budget again in the
first second of the next — twice the limit, back to back, which is exactly the burst
the limiter exists to stop.
"""

from __future__ import annotations

from paddock.api.ratelimit import RateLimiter


class FakeClock:
    """Monotonic seconds, moved by the test rather than by the machine."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter(limit: int = 3, window_s: float = 60.0) -> tuple[RateLimiter, FakeClock]:
    clock = FakeClock()
    return RateLimiter(limit=limit, window_s=window_s, clock=clock), clock


def test_calls_up_to_the_limit_are_allowed() -> None:
    limiter, _ = _limiter(limit=3)

    assert [limiter.check("1.1.1.1") for _ in range(3)] == [None, None, None]


def test_the_call_past_the_limit_is_refused() -> None:
    limiter, _ = _limiter(limit=3)
    for _ in range(3):
        limiter.check("1.1.1.1")

    assert limiter.check("1.1.1.1") is not None


def test_a_refused_call_does_not_extend_the_window() -> None:
    """Otherwise a caller who keeps hammering never gets back in."""
    limiter, clock = _limiter(limit=1, window_s=60.0)
    limiter.check("1.1.1.1")

    clock.advance(59.0)
    assert limiter.check("1.1.1.1") is not None

    clock.advance(1.5)
    assert limiter.check("1.1.1.1") is None


def test_the_window_slides_so_an_old_call_stops_counting() -> None:
    limiter, clock = _limiter(limit=2, window_s=60.0)
    limiter.check("1.1.1.1")
    clock.advance(30.0)
    limiter.check("1.1.1.1")

    assert limiter.check("1.1.1.1") is not None

    # The first call leaves the window; the second one, at +30 s, does not.
    clock.advance(31.0)
    assert limiter.check("1.1.1.1") is None
    assert limiter.check("1.1.1.1") is not None


def test_each_client_gets_its_own_budget() -> None:
    limiter, _ = _limiter(limit=1)
    assert limiter.check("1.1.1.1") is None

    assert limiter.check("2.2.2.2") is None
    assert limiter.check("1.1.1.1") is not None


def test_retry_after_is_when_the_oldest_call_leaves_the_window() -> None:
    limiter, clock = _limiter(limit=1, window_s=60.0)
    limiter.check("1.1.1.1")
    clock.advance(20.0)

    assert limiter.check("1.1.1.1") == 40


def test_retry_after_is_never_zero() -> None:
    """A `Retry-After: 0` invites an immediate retry that is refused again."""
    limiter, clock = _limiter(limit=1, window_s=60.0)
    limiter.check("1.1.1.1")
    clock.advance(59.99)

    assert limiter.check("1.1.1.1") == 1


def test_an_idle_client_is_forgotten() -> None:
    """The URL is public, so the key space is whatever the internet sends it.

    Without eviction a scan from a botnet is an unbounded dictionary on a box with
    12 GB and a 2.2 GB embedding model already resident.
    """
    limiter, clock = _limiter(limit=1, window_s=60.0)
    for octet in range(50):
        limiter.check(f"10.0.0.{octet}")
    assert limiter.tracked_clients == 50

    clock.advance(61.0)
    limiter.check("1.1.1.1")

    assert limiter.tracked_clients == 1


def test_a_limit_of_zero_refuses_everything() -> None:
    limiter, _ = _limiter(limit=0)

    assert limiter.check("1.1.1.1") is not None
