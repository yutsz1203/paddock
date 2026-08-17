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

from sqlalchemy import Select, func, literal, or_, select
from sqlalchemy.orm import Session, aliased

from paddock.db.models import Horse, HorseAlias, Jockey, Meeting, Race, Runner, Trainer

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

    # How the race was run, not just how it ended. Section count varies with
    # distance (3 at 1000m, 6 at 2400m), so these are arrays rather than fixed
    # columns — see the note on `Runner`.
    sectional_times: list[float] | None
    sectional_positions: list[int] | None


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
            sectional_times=runner.sectional_times,
            sectional_positions=runner.sectional_positions,
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


@dataclass(frozen=True)
class HorseMatch:
    """A horse named in a question, and the name that matched."""

    horse_id: str
    matched_name: str
    name_en: str | None
    name_zh: str | None


def find_horse(session: Session, question: str) -> HorseMatch | None:
    """Find the horse a question is about, by matching known names against the text.

    The direction is deliberately backwards from the obvious one. Extracting a name
    from free text needs a tokeniser that works in English and Chinese and knows that
    "GOLDEN SIXTY" is one name and "last start" is not. Asking the database which of
    its ~1,500 known names appears in this sentence needs no tokeniser at all, works
    in both languages, and can only ever return a horse that exists.

    Former names are matched too — HK horses are renamed mid-career, and a question
    asked with last season's name must not come back "no such horse" (T6).

    The longest match wins, so "TESTBRED FLYER" is preferred over a horse called
    "FLYER". That is the whole disambiguation: two horses whose names both appear in
    one sentence is a case for the full router (T16), not for this.
    """
    text = question.strip()
    if not text:
        return None

    asked = literal(text)
    current = session.execute(
        select(Horse.horse_id, Horse.name_en, Horse.name_zh).where(
            or_(
                Horse.name_en.is_not(None) & asked.ilike(func.concat("%", Horse.name_en, "%")),
                Horse.name_zh.is_not(None) & asked.ilike(func.concat("%", Horse.name_zh, "%")),
            )
        )
    ).all()
    former = session.execute(
        select(Horse.horse_id, Horse.name_en, Horse.name_zh, HorseAlias.name)
        .join(Horse, Horse.horse_id == HorseAlias.horse_id)
        .where(asked.ilike(func.concat("%", HorseAlias.name, "%")))
    ).all()

    # Postgres decided *whether* each row matches; this decides *which* name did, so
    # that the longest match can win. It runs over the handful of rows already
    # returned, not over the horses table.
    lowered = text.lower()
    candidates = [
        HorseMatch(horse_id=horse_id, matched_name=name, name_en=name_en, name_zh=name_zh)
        for horse_id, name_en, name_zh in current
        for name in (name_en, name_zh)
        if name and name.lower() in lowered
    ]
    candidates += [
        HorseMatch(horse_id=horse_id, matched_name=alias, name_en=name_en, name_zh=name_zh)
        for horse_id, name_en, name_zh, alias in former
    ]

    if not candidates:
        return None
    # `horse_id` breaks a tie on length. Without it `max` keeps whichever row Postgres
    # returned first, which is stable for an unchanged table and not stable across a
    # re-ingest or a vacuum — so the same question could resolve to a different horse
    # next week. The seeded meeting alone holds four seven-character names (MATZDEN,
    # STRAUSS, NUMBERS, SALON S), and a two-season backfill makes collisions ordinary.
    # An arbitrary-but-fixed tiebreak is not a disambiguation strategy; that is T16's
    # job. It is the difference between one wrong answer and an irreproducible one.
    return max(candidates, key=lambda match: (len(match.matched_name), match.horse_id))
