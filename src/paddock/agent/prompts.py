"""Prompts, versioned.

`PROMPT_VERSION` moves whenever any string here changes. It is stamped on every
trace (T18) so that an eval regression can be attributed to a specific edit rather
than to "something we changed last week" — which is the difference between a tuning
loop and a guessing loop.

Everything the model is told falls into three parts, and they are separated on
purpose: who it is and what the rules are (`SYSTEM`), what evidence it has and what
was asked (`question_prompt`), and — only when the first answer failed the citation
check — what specifically was wrong (`correction_prompt`).

The system prompt is short. A long one reads well and is followed less; the rules
that matter are the citation rule and the refusal rule, and burying them in a page
of tone guidance is how they stop being followed.
"""

from __future__ import annotations

from paddock.agent.citations import NO_EVIDENCE, Source, Verdict, format_sources

PROMPT_VERSION = "2026-08-16.1"

SYSTEM = f"""You answer questions about Hong Kong horse racing using only the \
evidence given to you.

Rules, in order of importance:

1. Every sentence that states a fact must cite its source with a marker like [S1]. \
Use the markers exactly as given. Never invent a marker.
2. Use only the evidence provided. You know nothing else about these horses, and \
anything you remember about racing is not evidence.
3. If the evidence does not answer the question, reply with exactly this and nothing \
else:
{NO_EVIDENCE}
4. Absence of a stewards' comment means the horse ran without incident. Say that \
plainly; do not invent an incident, and do not treat silence as evidence of trouble.
5. Be brief. Two or three sentences answers most questions.
6. You are not predicting anything. If you offer a lean, label it as a guess.

Answer in the language the question was asked in."""


def question_prompt(question: str, sources: list[Source]) -> str:
    """The evidence, then the question — in that order.

    Evidence first is deliberate: a model that has read the question before the
    sources starts composing an answer and then looks for support, which is exactly
    the failure the citation check exists to catch.
    """
    return f"""Evidence:
{format_sources(sources)}

Question: {question}"""


def correction_prompt(verdict: Verdict) -> str:
    """Name what failed. "Try again" gets the same answer back."""
    problems = []
    if verdict.unknown_markers:
        markers = ", ".join(f"[{marker}]" for marker in verdict.unknown_markers)
        problems.append(
            f"You cited {markers}, which does not exist. Only the markers listed under "
            "Evidence are real."
        )
    if verdict.uncited:
        quoted = "\n".join(f"  - {sentence}" for sentence in verdict.uncited)
        problems.append(f"These sentences state facts but cite nothing:\n{quoted}")

    return f"""Your answer was rejected.

{chr(10).join(problems)}

Rewrite it so that every factual sentence carries a marker from the Evidence above. \
If the evidence does not support a sentence, delete the sentence. If nothing in the \
evidence answers the question, reply with exactly:
{NO_EVIDENCE}"""
