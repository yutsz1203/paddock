"""Test doubles shared across the agent, API and ingestion tests.

These live here rather than in `src/` because nothing in production should be able
to import a fake LLM — or a fake HKJC — by accident.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence

import httpx

from paddock.llm.provider import Message

# Split into pieces that keep their trailing whitespace, so joining the stream is
# lossless — the same property the real providers have to preserve.
_PIECES = re.compile(r"\S+\s*")


class ScriptedLLM:
    """Replays fixed replies, one per call, streamed in word-sized pieces.

    Every agent test drives the graph with one of these: the questions being asked
    are about routing, citation enforcement and abstention, none of which should
    depend on what a model happens to say today.
    """

    name = "scripted"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if self.replies else ""
        yield from _PIECES.findall(reply)

    @property
    def prompts(self) -> list[str]:
        """Every message body sent so far, for asserting what the model was told."""
        return [message.content for call in self.calls for message in call]


class RecordingFetcher:
    """Serves canned pages by URL and records what was asked for.

    Stands in for `HkjcClient` wherever a test needs to prove something about
    *requests* rather than about parsing — that the archive is consulted first, that
    a failure part-way leaves the earlier pages fetched, that nothing re-fetches on a
    second run. Keyed by the same `path?query` string the real client builds, so a
    test that gets the key wrong fails loudly rather than serving the wrong page.
    """

    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.requests: list[str] = []
        self.fail_on: dict[str, Exception] = {}

    def serve(self, path: str, params: Mapping[str, str] | None, body: str) -> str:
        """Register a page and return the URL it will be requested under."""
        url = self.url_for(path, params)
        self.pages[url] = body
        return url

    def fail(self, path: str, params: Mapping[str, str] | None, error: Exception) -> None:
        """Make one URL raise, to prove what a mid-meeting failure leaves behind."""
        self.fail_on[self.url_for(path, params)] = error

    def url_for(self, path: str, params: Mapping[str, str] | None = None) -> str:
        # httpx's own encoder, not urlencode: it percent-encodes the slashes in
        # `date=2026/04/26`, and a double that encoded differently would key the page
        # archive differently from production.
        return f"{path}?{httpx.QueryParams(params)}" if params else path

    def get_text(self, path: str, params: Mapping[str, str] | None = None) -> str:
        url = self.url_for(path, params)
        self.requests.append(url)
        if url in self.fail_on:
            raise self.fail_on[url]
        try:
            return self.pages[url]
        except KeyError:
            raise AssertionError(
                f"no canned page for {url!r}; known: {sorted(self.pages)}"
            ) from None
