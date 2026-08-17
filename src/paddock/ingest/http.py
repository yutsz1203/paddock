"""Polite HTTP client for racing.hkjc.com.

One client for all ingestion, so the politeness guarantees are structural rather
than a convention each caller has to remember: a minimum delay between requests,
bounded retries with exponential backoff, and an honest User-Agent that identifies
the project.

The delay is enforced here rather than by callers because a caller that forgets is
indistinguishable from an attack.

**What is worth retrying is narrower than "anything that raised".** A 503 is HKJC
having a moment and the next attempt will probably work. A 404 will still be a 404
in eight seconds — and backfill generates candidate dates of which most are not
meetings, so retrying misses would triple the cost of the common case. 429 is
retried, because being told to slow down is exactly what backing off is for.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from paddock.config import Settings, get_settings

# Statuses where waiting changes the answer. Everything else is a fact about the
# request, not about the server's mood.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in _RETRY_STATUSES
    return False


class HkjcClient:
    """Rate-limited HTTP client. Use as a context manager.

    Args:
        settings: defaults to the process settings. Passed in so tests can shorten
            the delay and the backoff without touching the environment.
        transport: an `httpx` transport, so tests can answer requests without a
            network. Production leaves it None and gets the real one.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._delay_s = settings.hkjc_request_delay_s
        self._last_request_at = 0.0
        self._client = httpx.Client(
            base_url=settings.hkjc_base_url,
            headers={"User-Agent": settings.hkjc_user_agent},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            transport=transport,
        )
        # Built per instance rather than as a decorator, so the attempt count and the
        # backoff come from settings — a hardcoded policy makes them untestable and
        # makes `hkjc_max_retries` a lie.
        self._retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(settings.hkjc_max_retries),
            wait=wait_exponential(
                multiplier=settings.hkjc_retry_backoff_s,
                min=settings.hkjc_retry_backoff_s,
                max=30,
            ),
            reraise=True,
        )

    def __enter__(self) -> HkjcClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def url_for(self, path: str, params: Mapping[str, str] | None = None) -> str:
        """The absolute URL a `get_text(path, params)` would request.

        The archive keys on this, so it is built by the same client that would do the
        fetching — a separately assembled URL would drift and silently split one
        page's history into two.
        """
        return str(self._client.build_request("GET", path, params=params).url)

    def get_text(self, path: str, params: Mapping[str, str] | None = None) -> str:
        """GET `path`, returning the response body. Retries transport and 5xx errors.

        Note that HKJC returns 200 for dates that do not exist, serving the most
        recent meeting instead. No HTTP-level check can detect that — see
        `date_guard`, which is why it exists.
        """
        return str(self._retrying(self._get_once, path, params))

    def _get_once(self, path: str, params: Mapping[str, str] | None) -> str:
        self._wait_turn()
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.text

    def _wait_turn(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._delay_s:
            time.sleep(self._delay_s - elapsed)
        self._last_request_at = time.monotonic()
