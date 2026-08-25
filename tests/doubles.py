"""Test doubles shared across the agent, API and ingestion tests.

These live here rather than in `src/` because nothing in production should be able
to import a fake LLM — or a fake HKJC — by accident.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence

import httpx

from paddock.db.models import EMBEDDING_DIM
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


class FakeEmbedder:
    """Deterministic vectors from a hash — same text, same vector, no model.

    Distances between these are meaningless, which is the point: any test that needs
    real semantics must ask for the real model rather than quietly passing here.

    `fail_on` makes one text raise. Embedding is the one step in this project that
    takes hours, so what a run does when it dies half-way is a property worth a test,
    and the only cheap way to reach that state is to break the encoder on purpose.
    """

    dim = EMBEDDING_DIM

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail_on is not None and any(self.fail_on in text for text in texts):
            raise RuntimeError(f"encoder refused {self.fail_on!r}")
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(EMBEDDING_DIM)]

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]
