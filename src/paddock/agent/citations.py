"""Cite or abstain — the one rule the whole answer path is built around.

An answer about a horse that a reader cannot check is worse than no answer. The
model has read enough racing text to produce a fluent, plausible, entirely invented
account of a run, and nothing downstream would notice. So every factual sentence
must point at a row we retrieved, and an answer that cannot do that must say so.

## The check is mechanical, not judged

No LLM grades this. A model asked whether its own answer is grounded says yes; a
second model costs a call per request and is itself ungrounded. Instead the rule is
a string rule, which is crude in one direction only — it can reject a well-grounded
sentence that forgot its marker, and the graph handles that by regenerating once.
It cannot pass a claim that cites nothing, which is the direction that matters.

The rule: **every sentence carries at least one marker naming a retrieved source**,
except two shapes that assert nothing —

- a sentence ending in a colon, which introduces the evidence that follows;
- a question.

## An invented marker is the worst outcome

``[S9]`` when only S1 and S2 were retrieved looks resolvable, so a reader stops
checking. It is treated as a harder failure than a missing marker, and is reported
separately so the regeneration prompt can name it.

## Abstention is grounded

"I don't have evidence for that" is the one true sentence that cites nothing. It
passes by identity with `NO_EVIDENCE` rather than by pattern, so that an answer
merely *containing* the words "no evidence" is still checked normally — otherwise
the phrase becomes a way to smuggle uncited claims past the guard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["comment", "run"]

NO_EVIDENCE = (
    "I don't have evidence for that in the data I hold — "
    "it covers HKJC race meetings only, and nothing I retrieved is relevant."
)
"""The abstention. One exact string, so the check is identity and not a guess."""

_MARKER = re.compile(r"\[(S\d+)\]")

# Sentence-ish. Splitting on newlines as well as terminators keeps markdown bullets
# separate, so an uncited bullet cannot hide behind a cited neighbour.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# What is left after markers and punctuation are removed. A "sentence" that is only
# a marker asserts nothing and is not an answer.
_SUBSTANTIVE = re.compile(r"[A-Za-z一-鿿]")


@dataclass(frozen=True)
class Source:
    """One retrieved item, as the prompt and the answer will refer to it."""

    marker: str
    """``S1``, ``S2`` — assigned by us, never by the model."""
    kind: SourceKind
    text: str
    reference: str
    """Where it came from, e.g. ``incident_comment:77`` — what a source card resolves."""


@dataclass(frozen=True)
class Verdict:
    grounded: bool
    uncited: list[str]
    """Sentences that assert something and cite nothing."""
    unknown_markers: list[str]
    """Markers naming a source that was never retrieved."""
    abstained: bool = False


def number_sources(items: list[tuple[SourceKind, str, str]]) -> list[Source]:
    """Assign S1..Sn in the order given.

    Markers are ours, not the model's: it is asked to reuse them, and anything it
    invents fails `verify`. Numbering by position also makes a trace readable — S1 is
    always the first thing retrieved.
    """
    return [
        Source(marker=f"S{index}", kind=kind, text=text, reference=reference)
        for index, (kind, text, reference) in enumerate(items, start=1)
    ]


def format_sources(sources: list[Source]) -> str:
    """Render the evidence block for the prompt."""
    if not sources:
        # An empty block reads as a formatting bug and invites the model to fill the
        # gap from memory. Saying the gap is real is what makes abstention available.
        return "No sources were retrieved for this question."
    return "\n".join(f"[{source.marker}] {source.text}" for source in sources)


def verify(answer: str, sources: list[Source]) -> Verdict:
    """Check every sentence of `answer` against the sources it was given."""
    if answer.strip() == NO_EVIDENCE:
        return Verdict(grounded=True, uncited=[], unknown_markers=[], abstained=True)

    known = {source.marker for source in sources}
    text = answer.strip()
    if not text:
        return Verdict(grounded=False, uncited=[], unknown_markers=[])

    unknown = sorted(
        {marker for marker in _MARKER.findall(text) if marker not in known},
        key=lambda marker: int(marker[1:]),
    )

    uncited = [
        sentence
        for sentence in (part.strip() for part in _SENTENCE_SPLIT.split(text))
        if _needs_citation(sentence) and not _MARKER.search(sentence)
    ]

    # A "sentence" that is only markers asserts nothing, so a bare "[S1]" is not an
    # answer even though it technically cites.
    substantive = bool(_SUBSTANTIVE.search(_MARKER.sub("", text)))

    return Verdict(
        grounded=not uncited and not unknown and substantive,
        uncited=uncited,
        unknown_markers=unknown,
    )


def _needs_citation(sentence: str) -> bool:
    """False for the two shapes that assert nothing: a lead-in and a question."""
    stripped = sentence.strip()
    if not stripped or not _SUBSTANTIVE.search(_MARKER.sub("", stripped)):
        return False
    # The fullwidth pair is the same rule for a Chinese answer, not a typo.
    return not stripped.endswith((":", "?", "：", "？"))  # noqa: RUF001
