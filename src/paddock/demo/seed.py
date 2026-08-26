"""Cut the corpus down to the slice that ships in the repository.

`data/seed/paddock_demo.dump` lets a stranger clone the repository and ask a
question without scraping HKJC for an hour. That makes it the first thing anyone
runs and the last thing anyone reads, so what it holds has to be a decision rather
than whatever fitted.

**Why a slice and not the corpus.** Two reasons, and either alone would be enough.
Spec §436 forbids publishing bulk HKJC-derived data beyond a small demo slice. And
the vectors are 12.7 kB each as text: all 36,417 of them compress to about 200 MB,
which GitHub blocks. Twenty meetings compress to about 25 MB.

**What the slice drops, and why each one is a decision.**

*The page archive.* 117 MB of HKJC's own HTML. It is what makes a parser fix cheap
(T11) and it is exactly the bulk republication §436 rules out. `paddock check
integrity` re-derives each meeting's date from these pages, so it cannot run against
the demo database. That is stated in `data/seed/README.md` rather than worked around.

*Orphan chunks.* `chunks.source_id` carries no foreign key, so nothing in the
database removes a chunk when its comment is deleted. Left behind, such a chunk is a
vector that still answers a search and then cites a comment id that does not resolve
— the failure the citation rule exists to prevent.

*Horses, jockeys and trainers with nothing left to their name.* A horse row whose
every start was cut out answers "no runs found", which is a claim about form. A
horse that is absent answers "I do not know that horse", which is true.

*Ingest bookkeeping.* A restored snapshot never ran an ingest. A watermark left
behind tells a later `ingest since` that dates the demo does not hold were done.

## Not for the Oracle box

T24 restores the **whole** corpus onto the instance, including the page archive and
the watermarks, because that box runs the live pipeline from 6 September. That is a
plain `pg_dump` of the real database and needs no code — see `data/seed/README.md`.
This module builds the small public one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from paddock.db.models import (
    Chunk,
    FetchedPage,
    Horse,
    IncidentComment,
    IngestRun,
    Jockey,
    Meeting,
    Race,
    Runner,
    Trainer,
    Watermark,
)

DEFAULT_MEETINGS = 20
"""Meetings in the committed slice.

About 25 MB compressed, roughly five months of racing at both courses. Chosen
against two ceilings: GitHub warns at 50 MB and refuses at 100 MB, and a clone
should not cost more than the repository it carries.
"""

COMMENT_SOURCE = "incident_comment"
"""The only `chunks.source_type` in the corpus. Named here so the orphan sweep
deletes chunks it understands and leaves a future source type alone."""


@dataclass(frozen=True)
class SeedReport:
    """What a database holds, table by table.

    Printed by `paddock demo prune` and by `make demo` after the restore, because a
    dump is an opaque binary and the operator deserves to be told what came out of it.
    """

    meetings: int
    races: int
    runners: int
    comments: int
    chunks: int
    horses: int
    jockeys: int
    trainers: int
    pages: int
    first_date: dt.date | None
    last_date: dt.date | None


def seed_report(session: Session) -> SeedReport:
    """Count every table the slice touches. Reads only."""
    days = sorted(session.scalars(select(Meeting.race_date).distinct()))
    return SeedReport(
        meetings=_count(session, Meeting),
        races=_count(session, Race),
        runners=_count(session, Runner),
        comments=_count(session, IncidentComment),
        chunks=_count(session, Chunk),
        horses=_count(session, Horse),
        jockeys=_count(session, Jockey),
        trainers=_count(session, Trainer),
        pages=_count(session, FetchedPage),
        first_date=days[0] if days else None,
        last_date=days[-1] if days else None,
    )


def prune_to_recent_meetings(session: Session, *, meetings: int = DEFAULT_MEETINGS) -> SeedReport:
    """Delete everything outside the most recent `meetings` meetings.

    Destructive, and irreversibly so — run it against a copy of the corpus, never
    against the corpus. `make seed` makes that copy.

    Order matters. Meetings go first so the cascade takes their races, runners and
    comments; the orphan sweeps then run against what survived.
    """
    keep = list(
        session.scalars(
            select(Meeting.id).order_by(Meeting.race_date.desc(), Meeting.id.desc()).limit(meetings)
        )
    )
    session.execute(delete(Meeting).where(Meeting.id.not_in(keep)))
    session.flush()

    _delete_orphan_chunks(session)
    _delete_unraced_participants(session)

    # Bookkeeping for a scrape the demo did not do, and an archive it must not publish.
    session.execute(delete(FetchedPage))
    session.execute(delete(IngestRun))
    session.execute(delete(Watermark))
    session.flush()

    return seed_report(session)


def _delete_orphan_chunks(session: Session) -> None:
    """Drop chunks whose comment is gone. See the module docstring."""
    session.execute(
        delete(Chunk).where(
            Chunk.source_type == COMMENT_SOURCE,
            Chunk.source_id.not_in(select(IncidentComment.id)),
        )
    )
    session.flush()


def _delete_unraced_participants(session: Session) -> None:
    """Drop horses, jockeys and trainers that no surviving row names.

    A horse is named by a runner or by a comment; the two can differ, because a
    comment survives a race whose runner row was never filled in.
    """
    session.execute(
        delete(Horse).where(
            Horse.horse_id.not_in(select(Runner.horse_id)),
            Horse.horse_id.not_in(select(IncidentComment.horse_id)),
        )
    )
    session.execute(
        delete(Jockey).where(
            Jockey.id.not_in(select(Runner.jockey_id).where(Runner.jockey_id.is_not(None))),
            Jockey.id.not_in(
                select(IncidentComment.jockey_id).where(IncidentComment.jockey_id.is_not(None))
            ),
        )
    )
    session.execute(
        delete(Trainer).where(
            Trainer.id.not_in(select(Runner.trainer_id).where(Runner.trainer_id.is_not(None)))
        )
    )
    session.flush()


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0
