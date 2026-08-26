"""The client half of the `/ask` protocol.

`token` (repeatedly) → `sources` (once) → `done` (once), or `error` in place of
`done`. See `paddock.api.routes` for why the answer is verified before the first
token is sent.

## Why the sources are guarded

`st.write_stream` consumes the generator and then returns, and the source cards are
drawn after that. So the cards have to outlive the iteration, and reading them early
has to fail rather than return an empty list: an answer that cites [S1] rendered
with no card is exactly the unverifiable output this project refuses to produce.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

Event = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class SourceCard:
    """One citation, as the UI draws it."""

    marker: str
    kind: str
    text: str
    reference: str


def parse_sse(lines: Iterable[str]) -> Iterator[Event]:
    """Turn the lines of an SSE body into (event, payload) pairs.

    Args:
        lines: the response body split on newlines — what `httpx.iter_lines` gives.

    Yields:
        One pair per block. Blocks without an `event:` field, and comment lines
        beginning with `:`, are skipped: a keepalive is not an event.
    """
    name: str | None = None
    data: list[str] = []

    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if name is not None:
                yield name, _payload(data)
            name, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].lstrip())

    # A body that ends without its blank line still carries a complete block.
    if name is not None:
        yield name, _payload(data)


def _payload(data: list[str]) -> dict[str, Any]:
    if not data:
        return {}
    parsed: dict[str, Any] = json.loads("\n".join(data))
    return parsed


@dataclass
class AnswerStream:
    """One answer, read as it arrives.

    Iterate it to get the text piece by piece. Everything else — the cards, the
    route, whether it refused — is readable once iteration finishes.
    """

    events: Iterable[Event]

    route: str | None = None
    horse_id: str | None = None
    horse_name: str | None = None
    abstained: bool = False
    error: str | None = None

    _sources: list[SourceCard] = field(default_factory=list)
    _drained: bool = False

    def __iter__(self) -> Iterator[str]:
        for name, payload in self.events:
            if name == "token":
                yield str(payload.get("text", ""))
            elif name == "sources":
                self._sources = [SourceCard(**item) for item in payload.get("sources", [])]
            elif name == "done":
                self.route = payload.get("route")
                self.horse_id = payload.get("horse_id")
                self.horse_name = payload.get("horse_name")
                self.abstained = bool(payload.get("abstained", False))
            elif name == "error":
                self.error = str(payload.get("message", "unknown"))
                break

        self._drained = True

    @property
    def sources(self) -> list[SourceCard]:
        """The citations.

        Raises:
            RuntimeError: the stream has not been read to the end. Half a stream is
                not an answer, and an empty list here would render a cited answer as
                an uncited one.
        """
        if not self._drained:
            raise RuntimeError("read the stream to the end before rendering its sources")
        return self._sources
