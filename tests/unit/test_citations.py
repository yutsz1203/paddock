"""The guardrail: cite or abstain.

Spec §10 says no uncited factual claim, ever. That is a hard boundary, so the check
that enforces it is a pure function with no LLM in the loop — an LLM asked to grade
its own citations will pass itself.

The rule these tests pin down is deliberately mechanical: every sentence carries a
valid marker, except the two shapes that assert nothing — a sentence ending in a
colon (it introduces the evidence that follows) and a question. A marker naming a
source that was never retrieved is worse than no marker at all, because it looks
resolvable and is not.
"""

from __future__ import annotations

import pytest

from paddock.agent.citations import Source, format_sources, verify

SOURCES = [
    Source(
        marker="S1",
        kind="comment",
        text="Was hampered approaching the 800 Metres.",
        reference="incident_comment:77",
    ),
    Source(
        marker="S2",
        kind="run",
        text="2026-04-26 ST R5 1200m Class 4 — finished 4 of 12, beaten 2.5L, 6.5.",
        reference="race:10",
    ),
]


# ── The happy path ──────────────────────────────────────────────────────────────


def test_a_fully_cited_answer_passes() -> None:
    verdict = verify("SETANTA was hampered near the 800 [S1].", SOURCES)

    assert verdict.grounded
    assert verdict.uncited == []
    assert verdict.unknown_markers == []


def test_several_sentences_each_need_their_own_marker() -> None:
    answer = "It finished 4th of 12 [S2]. It was hampered near the 800 [S1]."

    assert verify(answer, SOURCES).grounded


def test_one_sentence_may_cite_several_sources() -> None:
    assert verify("It was hampered and still ran on [S1][S2].", SOURCES).grounded


def test_the_marker_need_not_end_the_sentence() -> None:
    """Mid-sentence attribution reads better in a long answer and is still a
    citation — the position carries no meaning."""
    assert verify("Per the stewards [S1], it was hampered near the 800.", SOURCES).grounded


# ── What gets rejected ──────────────────────────────────────────────────────────


def test_an_uncited_sentence_fails() -> None:
    verdict = verify("It was hampered near the 800 [S1]. It should win next time.", SOURCES)

    assert not verdict.grounded
    assert verdict.uncited == ["It should win next time."]


def test_a_marker_for_a_source_that_was_never_retrieved_fails() -> None:
    """The most dangerous failure: an answer that looks cited, whose citation
    resolves to nothing. Worse than no marker, because a reader stops checking."""
    verdict = verify("It won by three lengths [S9].", SOURCES)

    assert not verdict.grounded
    assert verdict.unknown_markers == ["S9"]


def test_an_empty_answer_is_not_grounded() -> None:
    assert not verify("   ", SOURCES).grounded


def test_nothing_is_grounded_when_nothing_was_retrieved() -> None:
    """With no evidence there is no valid marker, so every answer fails here and the
    graph is forced down the abstention path."""
    assert not verify("It was hampered near the 800 [S1].", []).grounded


# ── Sentences that assert nothing ───────────────────────────────────────────────


def test_a_colon_sentence_introduces_evidence_and_needs_no_marker() -> None:
    answer = "Two things stand out from its last start:\nIt was hampered near the 800 [S1]."

    assert verify(answer, SOURCES).grounded


def test_a_question_needs_no_marker() -> None:
    assert verify("Did it have an excuse? It was hampered [S1].", SOURCES).grounded


def test_a_bullet_list_is_checked_line_by_line() -> None:
    """Markdown lists are how the answer to Q4 will be shaped — one runner per line —
    so an uncited bullet must not hide behind a cited neighbour."""
    answer = "- Hampered near the 800 [S1]\n- Likely to improve next start"

    verdict = verify(answer, SOURCES)

    assert not verdict.grounded
    assert verdict.uncited == ["- Likely to improve next start"]


# ── Abstention ──────────────────────────────────────────────────────────────────


def test_the_abstention_sentinel_is_grounded_by_definition() -> None:
    """ "I have no evidence" is the one true claim that cites nothing. It has to pass,
    or the graph regenerates forever trying to cite the absence of a source."""
    from paddock.agent.citations import NO_EVIDENCE

    verdict = verify(NO_EVIDENCE, [])

    assert verdict.grounded
    assert verdict.abstained


def test_an_answer_is_not_an_abstention_just_because_it_says_no() -> None:
    verdict = verify("No incident was reported for this horse [S1].", SOURCES)

    assert verdict.grounded
    assert not verdict.abstained


# ── Rendering sources for the prompt ────────────────────────────────────────────


def test_sources_are_numbered_for_the_prompt() -> None:
    rendered = format_sources(SOURCES)

    assert "[S1]" in rendered
    assert "[S2]" in rendered
    assert "Was hampered approaching the 800 Metres." in rendered


def test_markers_are_assigned_in_order() -> None:
    from paddock.agent.citations import number_sources

    numbered = number_sources([("comment", "a", "incident_comment:1"), ("run", "b", "race:2")])

    assert [source.marker for source in numbered] == ["S1", "S2"]


def test_formatting_no_sources_says_so_rather_than_returning_nothing() -> None:
    """An empty evidence block in a prompt reads as a formatting bug to the model and
    invites it to fill the gap from memory. Say the gap is real."""
    rendered = format_sources([])

    assert rendered.strip() != ""
    assert "no" in rendered.lower()


@pytest.mark.parametrize("answer", ["[S1]", "[S1] [S2]"])
def test_a_bare_marker_is_not_an_answer(answer: str) -> None:
    assert not verify(answer, SOURCES).grounded
