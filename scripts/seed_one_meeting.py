"""Seed one meeting from committed fixtures, so Checkpoint A can be run.

**This is scaffolding, not T10.** The real pipeline — idempotent upserts,
watermarks, rate limiting, `ingest_runs` bookkeeping, and actual HTTP — is Task 10.
This script exists because Checkpoint A asks for "one question about one meeting
answered end to end", and that gate sits *before* the pipeline in the plan. It
deletes and rewrites the meeting on every run rather than upserting, touches no
network, and should be deleted once `paddock ingest meeting` exists.

What it does:

    fixture HTML → date guard → parse → rows → embed

The date guard runs even though the input is a local file. The guard is the one
component every ingestion path must pass through (ADR-002), and a seeding script
that skipped it would be modelling a pipeline we do not want.

Everything comes from `tests/fixtures/html/` — the same pages the parser tests run
against. The incident report carries the full field for all 11 races (142 runners,
114 of them commented), so races, runners and comments all come from that one page;
the results and sectional fixtures cover Race 1 only and enrich it with weights,
odds, times, margins and sectionals.

Usage::

    uv run python scripts/seed_one_meeting.py            # seed + embed
    uv run python scripts/seed_one_meeting.py --skip-embed
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from paddock.config import get_settings
from paddock.db.models import Chunk, IncidentComment, Meeting, Race, Runner
from paddock.db.session import session_scope
from paddock.embed.store import embed_meeting
from paddock.ingest.date_guard import require_genuine
from paddock.ingest.entities import resolve_horse, resolve_jockey, resolve_trainer
from paddock.ingest.incident_report import MeetingReport, RunnerReport, parse_meeting_report
from paddock.ingest.results import RaceResults, ResultRunner, parse_race_results
from paddock.ingest.sectionals import RunnerSectionals, parse_sectional_times

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "html"

# The one meeting with a complete fixture set. A dict rather than constants so a
# second meeting is a data change, not a code change.
MEETINGS = {
    dt.date(2026, 4, 26): {
        "racecourse": "ST",
        "report": "report_20260426_valid.html",
        # Results and sectionals were captured for Race 1 only.
        "results": {1: "results_20260426_ST_R1.html"},
        "sectionals": {1: "sectional_20260426_R1.html"},
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=dt.date.fromisoformat,
        default=dt.date(2026, 4, 26),
        help="meeting to seed (must have a fixture set)",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="skip embedding — seeds rows only, so /ask cannot retrieve comments",
    )
    args = parser.parse_args()

    meeting_spec = MEETINGS.get(args.date)
    if meeting_spec is None:
        known = ", ".join(str(day) for day in MEETINGS)
        print(f"no fixture set for {args.date}. Known meetings: {known}", file=sys.stderr)
        return 1

    html = (FIXTURES / str(meeting_spec["report"])).read_text()

    # The guard, on a local file. Pointless here by construction — and that is the
    # point: no path to the database skips it.
    require_genuine(html, args.date)

    report = parse_meeting_report(html, args.date)
    results = {
        race_no: parse_race_results((FIXTURES / name).read_text())
        for race_no, name in dict(meeting_spec["results"]).items()
    }
    sectionals = {
        race_no: parse_sectional_times((FIXTURES / name).read_text())
        for race_no, name in dict(meeting_spec["sectionals"]).items()
    }

    with session_scope() as session:
        _delete_meeting(session, args.date, str(meeting_spec["racecourse"]))
        meeting_id = _write(
            session,
            race_date=args.date,
            racecourse=str(meeting_spec["racecourse"]),
            report=report,
            results=results,
            sectionals=sectionals,
        )

    counts = _counts(meeting_id)
    print(
        f"seeded {args.date} {meeting_spec['racecourse']}: "
        f"{counts['races']} races, {counts['runners']} runners, {counts['comments']} comments"
    )

    if args.skip_embed:
        print("skipped embedding — /ask will abstain on every question")
        return 0

    from paddock.embed.embedder import get_embedder  # heavy import; only when embedding

    print("embedding (bge-m3 loads on first use, ~1 minute)...")
    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=get_embedder())
    print(
        f"embedded {result.embedded} chunks of {result.total} "
        f"from {result.comments} comments ({result.unchanged} unchanged)"
    )

    print("\nnext:")
    print("  uv run uvicorn paddock.api.main:app --port 8000")
    print("  curl -N -X POST localhost:8000/ask -H 'content-type: application/json' \\")
    print("""    -d '{"question":"Did MATZDEN have any trouble in running last time?"}'""")
    return 0


# ── Writing ─────────────────────────────────────────────────────────────────────


def _write(
    session: Session,
    *,
    race_date: dt.date,
    racecourse: str,
    report: MeetingReport,
    results: dict[int, RaceResults],
    sectionals: dict[int, list[RunnerSectionals]],
) -> int:
    settings = get_settings()
    report_url = (
        f"{settings.hkjc_base_url}/en-us/local/information/racereportfull?date={race_date:%Y/%m/%d}"
    )
    fetched_at = dt.datetime.now(dt.UTC)

    # Meeting-level going is only known from a results page; the incident report
    # does not carry it.
    first_result = next(iter(results.values()), None)

    meeting = Meeting(
        race_date=race_date,
        racecourse=racecourse,
        going=first_result.going if first_result else None,
        source_url=report_url,
        fetched_at=fetched_at,
    )
    session.add(meeting)
    session.flush()

    for race_report in report.races:
        result = results.get(race_report.race_no)
        race = Race(
            meeting_id=meeting.id,
            race_no=race_report.race_no,
            name=race_report.name,
            race_class=race_report.race_class,
            distance_m=race_report.distance_m,
            track=result.track if result else None,
            course=result.course if result else None,
            going=result.going if result else None,
            prize=result.prize if result else None,
            # Every runner on this page has a finishing position, so the meeting is
            # run. T13 is what makes 'declared' reachable.
            status="finished",
        )
        session.add(race)
        session.flush()

        by_horse = {r.horse_id: r for r in result.runners if r.horse_id} if result else {}
        by_brand = {s.brand_no: s for s in sectionals.get(race_report.race_no, [])}

        for runner_report in race_report.runners:
            _write_runner(
                session,
                race_id=race.id,
                race_date=race_date,
                report=runner_report,
                result=by_horse.get(runner_report.horse_id or ""),
                sectional=by_brand.get(runner_report.brand_no),
                source_url=report_url,
                fetched_at=fetched_at,
            )

    return meeting.id


def _write_runner(
    session: Session,
    *,
    race_id: int,
    race_date: dt.date,
    report: RunnerReport,
    result: ResultRunner | None,
    sectional: RunnerSectionals | None,
    source_url: str,
    fetched_at: dt.datetime,
) -> None:
    if report.horse_id is None:
        # No stable id means no join key. The brand number alone lacks the import
        # year, so inventing an id here would risk merging two eras of one brand.
        return

    resolve_horse(
        session,
        horse_id=report.horse_id,
        brand_no=report.brand_no,
        name_en=report.horse_name,
        seen_on=race_date,
    )
    jockey = resolve_jockey(session, name_en=report.jockey)
    trainer = (
        resolve_trainer(session, name_en=result.trainer)
        if result is not None and result.trainer
        else None
    )

    session.add(
        Runner(
            race_id=race_id,
            horse_id=report.horse_id,
            horse_no=report.horse_no,
            draw=report.draw,
            jockey_id=jockey.id,
            trainer_id=trainer.id if trainer else None,
            jockey_claim=report.jockey_claim,
            carried_weight_lb=result.carried_weight_lb if result else None,
            declared_horse_weight_lb=result.declared_horse_weight_lb if result else None,
            finish_pos=report.finish_pos,
            finish_time_s=result.finish_time_s if result else None,
            margin=result.margin if result else None,
            win_odds=result.win_odds if result else None,
            sectional_times=sectional.sectional_times if sectional else None,
            sectional_positions=sectional.running_positions if sectional else None,
        )
    )

    # Absence is meaningful: a runner without a comment ran clean, and gets no row
    # rather than an empty one (T4).
    if report.comment:
        session.add(
            IncidentComment(
                race_id=race_id,
                horse_id=report.horse_id,
                jockey_id=jockey.id,
                finish_pos=report.finish_pos,
                text_en=report.comment,
                source_url=source_url,
                fetched_at=fetched_at,
            )
        )


# ── Bookkeeping ─────────────────────────────────────────────────────────────────


def _delete_meeting(session: Session, race_date: dt.date, racecourse: str) -> None:
    """Delete and rewrite, rather than upsert. T10 is where idempotency lives."""
    meeting_id = session.scalar(
        select(Meeting.id).where(Meeting.race_date == race_date, Meeting.racecourse == racecourse)
    )
    if meeting_id is None:
        return

    race_ids = list(session.scalars(select(Race.id).where(Race.meeting_id == meeting_id)))
    comment_ids = list(
        session.scalars(select(IncidentComment.id).where(IncidentComment.race_id.in_(race_ids)))
    )
    # Chunks have no foreign key to comments — source_id is a plain int — so they
    # do not cascade and must go first.
    if comment_ids:
        session.query(Chunk).filter(Chunk.source_id.in_(comment_ids)).delete(
            synchronize_session=False
        )
    session.query(Meeting).filter(Meeting.id == meeting_id).delete(synchronize_session=False)
    session.flush()


def _counts(meeting_id: int) -> dict[str, int]:
    with session_scope() as session:
        race_ids = list(session.scalars(select(Race.id).where(Race.meeting_id == meeting_id)))
        return {
            "races": len(race_ids),
            "runners": len(
                list(session.scalars(select(Runner.id).where(Runner.race_id.in_(race_ids))))
            ),
            "comments": len(
                list(
                    session.scalars(
                        select(IncidentComment.id).where(IncidentComment.race_id.in_(race_ids))
                    )
                )
            ),
        }


if __name__ == "__main__":
    raise SystemExit(main())
