"""Form, as a query.

"How has this horse gone over 1200m at Sha Tin in its last 5 runs?" is not a
retrieval question. It is `WHERE distance_m = 1200 AND racecourse = 'ST' ORDER BY
race_date DESC LIMIT 5`, and no embedding expresses it — a vector search over form
lines will happily return the sixth run, or a 1400m run, with high confidence.
Knowing when *not* to retrieve is the engineering claim of this project (spec §1),
and this module is the half of the pair that makes the claim true.

## The agent picks a query; it never writes one

Every function here takes typed arguments and builds its statement through
SQLAlchemy, so values reach Postgres as bound parameters. There is no entry point
that accepts SQL text, and the agent's tool schema exposes only these signatures.
A horse name lifted straight out of a user prompt is therefore data, not code — the
injection tests in `test_retrieval.py` assert exactly that, because "the LLM would
never emit that" is not a security control.

## What counts as a run

Two exclusions, both of which would otherwise report a horse's form wrongly rather
than incompletely:

- **Declared but unrun races.** Runners exist from the moment the card is published,
  with NULL results (T2). Counting one as a run reports a horse as unplaced in a race
  that has not happened.
- **Scratchings.** A withdrawn horse never left the barrier. It is a fact about the
  race, not about the horse's form.

## Field size travels with the placing

"4th" alone is not a form line. Fourth of five and fourth of fourteen are different
facts, and whichever the tool returns is the one the agent will quote — so the count
of non-scratched runners comes back with every row rather than being left for a
caller to remember to fetch.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, aliased

from paddock.db.models import Jockey, Meeting, Race, Runner, Trainer

MAX_RUNS = 50
"""A hard ceiling on any one call. The agent chooses `n`, and an unbounded `n` from a
model that miscounted would drag a horse's entire career into the prompt."""


@dataclass(frozen=True)
class RunLine:
    """One past run, as it would appear in a form guide."""

    race_id: int
    race_date: dt.date
    racecourse: str
    """Venue — ST or HV."""
    race_no: int
    race_class: str | None
    distance_m: int | None
    course: str | None
    """Rail configuration — A, B, C+3 — not the venue."""
    going: str | None
    draw: int | None
    carried_weight_lb: int | None
    jockey: str | None
    trainer: str | None
    finish_pos: int | None
    field_size: int
    margin: float | None
    win_odds: float | None
    finish_time_s: float | None


def recent_runs(session: Session, *, horse_id: str, n: int = 5) -> list[RunLine]:
    """The horse's last `n` completed runs, newest first.

    Args:
        session: an open session.
        horse_id: HKJC's identifier, e.g. ``HK_2024_K570``. An unknown one returns
            no rows — a debutant has no form, which is an answer, not an error.
        n: how many runs, 1..`MAX_RUNS`.

    Raises:
        ValueError: `n` is outside 1..`MAX_RUNS`.
    """
    if not 1 <= n <= MAX_RUNS:
        raise ValueError(f"n must be between 1 and {MAX_RUNS}, got {n}")

    rows = session.execute(_form_query(horse_id).limit(n)).all()
    return [
        RunLine(
            race_id=race.id,
            race_date=meeting.race_date,
            racecourse=meeting.racecourse,
            race_no=race.race_no,
            race_class=race.race_class,
            distance_m=race.distance_m,
            course=race.course,
            going=race.going or meeting.going,
            draw=runner.draw,
            carried_weight_lb=runner.carried_weight_lb,
            jockey=jockey_name,
            trainer=trainer_name,
            finish_pos=runner.finish_pos,
            field_size=field_size,
            margin=runner.margin,
            win_odds=runner.win_odds,
            finish_time_s=runner.finish_time_s,
        )
        for runner, race, meeting, jockey_name, trainer_name, field_size in rows
    ]


def _form_query(horse_id: str) -> Select[tuple[Runner, Race, Meeting, str | None, str | None, int]]:
    """Completed runs for one horse, newest first, with the field size alongside.

    The field size is a correlated subquery rather than a join with GROUP BY: the
    grouping would have to repeat every selected column, and adding one column later
    is exactly the kind of edit that silently changes a GROUP BY's meaning.
    """
    rival = aliased(Runner)
    field_size = (
        select(func.count())
        .select_from(rival)
        .where(rival.race_id == Runner.race_id, rival.scratched.is_(False))
        .correlate(Runner)
        .scalar_subquery()
    )

    return (
        select(Runner, Race, Meeting, Jockey.name_en, Trainer.name_en, field_size)
        .join(Race, Race.id == Runner.race_id)
        .join(Meeting, Meeting.id == Race.meeting_id)
        .outerjoin(Jockey, Jockey.id == Runner.jockey_id)
        .outerjoin(Trainer, Trainer.id == Runner.trainer_id)
        .where(
            Runner.horse_id == horse_id,
            Runner.scratched.is_(False),
            # The result columns are NULL until the race is run, so this is what
            # separates form from a declaration — not `races.status`, which a
            # half-ingested meeting could leave stale.
            Runner.finish_pos.is_not(None),
        )
        .order_by(Meeting.race_date.desc(), Race.race_no.desc())
    )
