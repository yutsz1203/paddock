"""A per-client sliding-window limiter for `/ask`.

## Why a limiter at all

`/ask` is the expensive endpoint. One question is a query embedding, a vector
search and one or two model calls against a metered key, on a two-core box. The
public URL has no login, so the only thing between a scraper and the month's budget
is this module and the daily cap beside it in `paddock.api.budget`.

The two are different guards and both are needed. This one paces a single caller
so nobody starves the rest; the cap bounds the total spend however many callers
there are.

## Sliding window, not fixed

A fixed window lets a caller spend the whole budget in the last second of one
window and the whole budget again in the first second of the next — twice the limit,
back to back, which is the burst the limiter exists to stop. Keeping the timestamps
costs one small deque per active client and answers `Retry-After` exactly.

## In-process, and that is a decision

The counts live in memory, so they are per worker and they are lost on restart.
That is right for this deployment and wrong for a larger one. The API runs as a
single uvicorn process on one box (spec §147), so "per worker" and "per box" are
the same thing, and a shared store would be a Redis container bought to hold ten
integers. If the deployment ever grows a second worker, this becomes a Redis
counter and the interface does not change.

## What counts as a client

Behind Caddy every request arrives from the proxy, so the peer address is the same
for everybody and the limiter would meter the world as one caller. `client_key`
reads the forwarded address instead — but only as many hops as are configured,
because `X-Forwarded-For` is caller-supplied and a limiter that trusts it can be
defeated by one header.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from time import monotonic

from fastapi import Request


class RateLimiter:
    """How many calls one client may make in a window, and when it may call again."""

    def __init__(
        self,
        *,
        limit: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._calls: dict[str, deque[float]] = {}
        self._last_sweep = clock()

    def check(self, key: str) -> int | None:
        """Record a call and return None, or refuse and return the seconds to wait.

        Args:
            key: what identifies the caller — see `client_key`.

        Returns:
            None if the call is allowed. Otherwise whole seconds until the oldest
            recorded call leaves the window, which is the earliest a retry can
            succeed. Never zero: `Retry-After: 0` invites an immediate retry that is
            refused again.

        A refused call is **not** recorded. Recording it would push the window
        forward on every rejected attempt, so a caller who keeps hammering would
        never be let back in — a limiter that turns into a ban.
        """
        now = self._clock()
        self._sweep(now)

        if self.limit <= 0:
            return math.ceil(self.window_s)

        recent = self._calls.setdefault(key, deque())
        self._expire(recent, now)

        if len(recent) >= self.limit:
            return max(1, math.ceil(recent[0] + self.window_s - now))

        recent.append(now)
        return None

    @property
    def tracked_clients(self) -> int:
        """How many clients hold memory. Read by the eviction test, and by nothing else."""
        return len(self._calls)

    def _expire(self, calls: deque[float], now: float) -> None:
        cutoff = now - self.window_s
        while calls and calls[0] <= cutoff:
            calls.popleft()

    def _sweep(self, now: float) -> None:
        """Drop clients that have gone quiet for a whole window.

        The URL is public, so the key space is whatever the internet sends it. A
        scan from many addresses would otherwise grow an unbounded dictionary on a
        box that already holds a 2.2 GB embedding model.

        Swept once per window rather than on every call, so a burst from many
        addresses does not make each request walk the whole table.
        """
        if now - self._last_sweep < self.window_s:
            return
        self._last_sweep = now

        for key in list(self._calls):
            self._expire(self._calls[key], now)
            if not self._calls[key]:
                del self._calls[key]


def client_key(request: Request, *, trusted_proxy_hops: int) -> str:
    """The address to meter this request against.

    Args:
        request: the incoming request.
        trusted_proxy_hops: how many reverse proxies stand in front of the API. Zero
            — the local default — ignores `X-Forwarded-For` entirely.

    Returns:
        The caller's address, or `"unknown"` if the peer address is missing, which
        happens under some ASGI test transports.

    Each trusted proxy **appends** the address it saw, so the rightmost entry is the
    one the nearest proxy wrote and the leftmost entries are whatever the caller
    chose to send. Counting from the right by exactly the number of proxies we run
    is what makes the header safe to read: a caller who sends
    `X-Forwarded-For: 1.2.3.4` gets their real address appended after it by Caddy,
    and it is that appended entry we take.

    The second half of the defence is in the Compose file, not here: the API port is
    not published, so the only host that can reach it is Caddy.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return peer

    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",")]
    forwarded = [part for part in forwarded if part]
    if len(forwarded) < trusted_proxy_hops:
        # Fewer entries than proxies means the chain is not what we configured. Fall
        # back to the peer rather than trust an entry a caller could have written.
        return peer
    return forwarded[-trusted_proxy_hops]
