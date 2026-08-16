"""Polite HTTP client for racing.hkjc.com.

One client for all ingestion, so the politeness guarantees are structural rather
than a convention each caller has to remember: a minimum delay between requests,
bounded retries with exponential backoff, and an honest User-Agent that identifies
the project.

The delay is enforced here rather than by callers because a caller that forgets is
indistinguishable from an attack.
"""

from __future__ import annotations

import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from paddock.config import get_settings


class HkjcClient:
    """Rate-limited HTTP client. Use as a context manager."""

    def __init__(self) -> None:
        settings = get_settings()
        self._delay_s = settings.hkjc_request_delay_s
        self._last_request_at = 0.0
        self._client = httpx.Client(
            base_url=settings.hkjc_base_url,
            headers={"User-Agent": settings.hkjc_user_agent},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )

    def __enter__(self) -> HkjcClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _wait_turn(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._delay_s:
            time.sleep(self._delay_s - elapsed)
        self._last_request_at = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def get_text(self, path: str, params: dict[str, str] | None = None) -> str:
        """GET `path`, returning the response body. Retries transport and 5xx errors.

        Note that HKJC returns 200 for dates that do not exist, serving the most
        recent meeting instead. No HTTP-level check can detect that — see
        `date_guard`, which is why it exists.
        """
        self._wait_turn()
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.text
