"""Test doubles shared across the agent and API tests.

These live here rather than in `src/` because nothing in production should be able
to import a fake LLM by accident.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

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
