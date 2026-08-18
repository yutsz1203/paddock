"""Ingestion orchestration: bookkeeping, idempotent upserts, atomicity.

The parsers (T4, T5) are pure functions over a page body and the archive (`pages.py`)
supplies that body without hitting the network twice. What is left is the part that
touches the database, and it has to hold three guarantees at once:

**Ingesting a meeting twice changes nothing the second time.** Every write is an
upsert keyed on the natural uniqueness the schema already declares — `(race_date,
racecourse)`, `(meeting_id, race_no)`, `(race_id, horse_id)`. Not "delete then
rewrite", which would churn primary keys and orphan the chunks that cite them.

**A meeting that fails half-way leaves nothing behind.** One transaction per meeting.
Nine races written and the tenth malformed means zero races written.

**But the bookkeeping survives that rollback.** The run record and the archived pages
are written in their own transactions, so a failure is visible afterwards and the
pages it cost are not paid for again. Bookkeeping that rolls back with the work it
describes would only ever record successes.

## Shape of one meeting

    report page ─ guard ─ parse ─┐
                                 ├─ one transaction ─ upserts ─ watermark
    per race: results, sectionals┘

The per-race pages are fetched *inside* the meeting transaction rather than gathered
first. That holds the transaction open for the length of the fetches — 22 requests at
1 req/s for a full card — which is acceptable here because ingestion is a single
writer with no contention, and it is what makes "failed half-way" a real state to
test rather than an artifact of doing all the reads before all the writes. The pages
themselves are committed independently, so nothing fetched is lost to the rollback.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from paddock.db.models import IncidentComment, IngestRun, Meeting, Race, Runner
from paddock.db.session import session_scope
from paddock.ingest.date_guard import FallbackDetectedError, require_genuine
from paddock.ingest.entities import resolve_horse, resolve_jockey, resolve_trainer
from paddock.ingest.incident_report import (
    RaceReport,
    ReportParseError,
    RunnerReport,
    parse_meeting_report,
)
from paddock.ingest.pages import PageFetcher, fetch_page
from paddock.ingest.results import (
    RaceResults,
    ResultRunner,
    ResultsParseError,
    parse_race_results,
)
from paddock.ingest.sectionals import (
    RunnerSectionals,
    SectionalParseError,
    parse_sectional_times,
    sectional_date_param,
)
from paddock.ingest.watermark import INCIDENT_REPORT, advance_watermark

REPORT_PATH = "/en-us/local/information/racereportfull"

# The legacy page, because the modern `localresults` builds its tables in the browser
# and serves none of them to a plain fetch (see `results.py`).
RESULTS_PATH = "/racing/information/English/Racing/LocalResults.aspx"
SECTIONALS_PATH = "/en-us/local/information/displaysectionaltime"


def report_params(race_date: dt.date) -> dict[str, str]:
    return {"date": race_date.strftime("%Y/%m/%d")}


def results_params(race_date: dt.date, racecourse: str, race_no: int) -> dict[str, str]:
    return {
        "RaceDate": race_date.strftime("%Y/%m/%d"),
        "Racecourse": racecourse,
        "RaceNo": str(race_no),
    }


def sectionals_params(race_date: dt.date, race_no: int) -> dict[str, str]:
    # Day-first here and nowhere else — see `sectionals.sectional_date_param`.
    return {"racedate": sectional_date_param(race_date), "RaceNo": str(race_no)}


@dataclass(frozen=True)
class MeetingIngest:
    meeting_id: int
    races: int
    runners: int
    comments: int
    races_without_results: list[int]
    """Races the results page had nothing for. Empty on a healthy meeting; T11's
    backfill logs these rather than discarding the meeting over them."""


def ingest_meeting(
    client: PageFetcher,
    race_date: dt.date,
    racecourse: str | None = None,
    *,
    refresh: bool = False,
) -> MeetingIngest:
    """Ingest one meeting end to end, idempotently and atomically.

    Args:
        racecourse: 'ST' or 'HV'. Optional because backfill (T11) works from a list
            of dates and has no venue to pass — left out, it is read from the
            report page's going table.
        refresh: re-fetch every page instead of reading the archive. For a meeting
            whose stewards' report HKJC corrected after publication.

    Raises:
        FallbackDetectedError: the date has no meeting and HKJC served another one.
        ReportParseError: no racecourse was given and the page did not name one.
        ResultsParseError: no race on the card had results, which means the endpoint
            changed rather than that one page was missing.
    """
    with record_run(INCIDENT_REPORT, race_date):
        report_page = fetch_page(client, REPORT_PATH, report_params(race_date), refresh=refresh)

        # Before parsing anything: the report endpoint answers 200 with the most
        # recent meeting for a date that never raced (ADR-002).
        require_genuine(report_page.body, race_date)
        report = parse_meeting_report(report_page.body, race_date)

        course = racecourse or report.racecourse
        if course is None:
            # Defaulting to Sha Tin would be right roughly half the time, and the
            # other half would fetch eleven empty results pages and store a card
            # with no result on it — which reads as a successful ingest.
            raise ReportParseError(
                f"the report for {race_date.isoformat()} named no racecourse and none "
                "was given; pass --course"
            )

        with session_scope() as session:
            return _write_meeting(
                session,
                client,
                race_date=race_date,
                racecourse=course,
                report_races=report.races,
                source_url=report_page.url,
                fetched_at=report_page.fetched_at,
                refresh=refresh,
            )


def _write_meeting(
    session: Session,
    client: PageFetcher,
    *,
    race_date: dt.date,
    racecourse: str,
    report_races: list[RaceReport],
    source_url: str,
    fetched_at: dt.datetime,
    refresh: bool,
) -> MeetingIngest:
    """The whole meeting, in one transaction. Any exception leaves nothing behind."""
    results: dict[int, RaceResults] = {}
    sectionals: dict[int, list[RunnerSectionals]] = {}
    missing: list[int] = []

    for race in report_races:
        parsed = _fetch_results(client, race_date, racecourse, race.race_no, refresh=refresh)
        if parsed is None:
            missing.append(race.race_no)
        else:
            results[race.race_no] = parsed
        sectionals[race.race_no] = _fetch_sectionals(
            client, race_date, race.race_no, refresh=refresh
        )

    if missing and len(missing) == len(report_races):
        # One missing results page is a gap; every one missing is the endpoint having
        # moved, and ingesting the card without a single finishing time would look
        # like a successful meeting for two seasons.
        raise ResultsParseError(
            f"no race on {race_date} had a results page — the endpoint has changed"
        )

    # Meeting-level going comes from a results page; the incident report has none.
    first_result = next(iter(results.values()), None)
    meeting_id = _upsert_meeting(
        session,
        race_date=race_date,
        racecourse=racecourse,
        going=first_result.going if first_result else None,
        source_url=source_url,
        fetched_at=fetched_at,
    )

    runners = comments = 0
    for race in report_races:
        result = results.get(race.race_no)
        race_id = _upsert_race(session, meeting_id=meeting_id, race=race, result=result)

        by_horse = {r.horse_id: r for r in result.runners if r.horse_id} if result else {}
        by_brand = {s.brand_no: s for s in sectionals.get(race.race_no, [])}

        for runner in race.runners:
            written = _upsert_runner(
                session,
                race_id=race_id,
                race_date=race_date,
                report=runner,
                result=by_horse.get(runner.horse_id or ""),
                sectional=by_brand.get(runner.brand_no),
                source_url=source_url,
                fetched_at=fetched_at,
            )
            runners += written.runner
            comments += written.comment

    advance_watermark(session, INCIDENT_REPORT, race_date)

    return MeetingIngest(
        meeting_id=meeting_id,
        races=len(report_races),
        runners=runners,
        comments=comments,
        races_without_results=missing,
    )


def _fetch_results(
    client: PageFetcher, race_date: dt.date, racecourse: str, race_no: int, *, refresh: bool
) -> RaceResults | None:
    """One race's results, or None when the page carried none.

    A results page with no table is this endpoint's honest way of saying it has
    nothing (unlike the report endpoint, which substitutes another meeting). One race
    missing is tolerated because the incident report already carries the card, and
    every result column is nullable by design; the caller decides what a *whole*
    meeting of them means.
    """
    page = fetch_page(
        client, RESULTS_PATH, results_params(race_date, racecourse, race_no), refresh=refresh
    )
    try:
        return parse_race_results(page.body)
    except ResultsParseError:
        return None


def _fetch_sectionals(
    client: PageFetcher, race_date: dt.date, race_no: int, *, refresh: bool
) -> list[RunnerSectionals]:
    """Per-section times, or nothing. HKJC publishes these later than the results."""
    page = fetch_page(
        client, SECTIONALS_PATH, sectionals_params(race_date, race_no), refresh=refresh
    )
    try:
        return parse_sectional_times(page.body)
    except SectionalParseError:
        return []


# ── Upserts ─────────────────────────────────────────────────────────────────────
#
# Each one is `INSERT … ON CONFLICT DO UPDATE` against the uniqueness the schema
# already declares, returning the existing primary key on a conflict. Delete-then-
# rewrite would be simpler to write and would reassign every id — `chunks.source_id`
# cites `incident_comments.id` without a foreign key to stop it.


def _upsert_meeting(
    session: Session,
    *,
    race_date: dt.date,
    racecourse: str,
    going: str | None,
    source_url: str,
    fetched_at: dt.datetime,
) -> int:
    statement = insert(Meeting).values(
        race_date=race_date,
        racecourse=racecourse,
        going=going,
        source_url=source_url,
        fetched_at=fetched_at,
    )
    return _upsert(
        session,
        statement,
        index_elements=["race_date", "racecourse"],
        update={"going": going, "source_url": source_url, "fetched_at": fetched_at},
        primary_key=Meeting.id,
    )


def _upsert_race(
    session: Session, *, meeting_id: int, race: RaceReport, result: RaceResults | None
) -> int:
    values = {
        "meeting_id": meeting_id,
        "race_no": race.race_no,
        "name": race.name,
        "race_class": race.race_class,
        "distance_m": race.distance_m,
        "track": result.track if result else None,
        "course": result.course if result else None,
        "going": result.going if result else None,
        "prize": result.prize if result else None,
        # Every runner on an incident report has already run. T13 is what makes
        # 'declared' reachable, by ingesting the card before the meeting.
        "status": "finished",
    }
    return _upsert(
        session,
        insert(Race).values(**values),
        index_elements=["meeting_id", "race_no"],
        update={k: v for k, v in values.items() if k not in ("meeting_id", "race_no")},
        primary_key=Race.id,
    )


@dataclass(frozen=True)
class _Written:
    runner: int
    comment: int


def _upsert_runner(
    session: Session,
    *,
    race_id: int,
    race_date: dt.date,
    report: RunnerReport,
    result: ResultRunner | None,
    sectional: RunnerSectionals | None,
    source_url: str,
    fetched_at: dt.datetime,
) -> _Written:
    if report.horse_id is None:
        # No stable id means no join key. The brand number alone lacks the import
        # year, so inventing an id here risks merging two eras of one brand.
        return _Written(runner=0, comment=0)

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

    values = {
        "race_id": race_id,
        "horse_id": report.horse_id,
        "horse_no": report.horse_no,
        "draw": report.draw,
        "jockey_id": jockey.id,
        "trainer_id": trainer.id if trainer else None,
        "jockey_claim": report.jockey_claim,
        "carried_weight_lb": result.carried_weight_lb if result else None,
        "declared_horse_weight_lb": result.declared_horse_weight_lb if result else None,
        "finish_pos": report.finish_pos,
        "finish_time_s": result.finish_time_s if result else None,
        "margin": result.margin if result else None,
        "win_odds": result.win_odds if result else None,
        "sectional_times": sectional.sectional_times if sectional else None,
        "sectional_positions": sectional.running_positions if sectional else None,
    }
    _upsert(
        session,
        insert(Runner).values(**values),
        index_elements=["race_id", "horse_id"],
        update={k: v for k, v in values.items() if k not in ("race_id", "horse_id")},
        primary_key=Runner.id,
    )

    # Absence is meaningful: a runner without a comment ran clean, and gets no row
    # rather than an empty one (T4).
    if not report.comment:
        return _Written(runner=1, comment=0)

    comment_values = {
        "race_id": race_id,
        "horse_id": report.horse_id,
        "jockey_id": jockey.id,
        "finish_pos": report.finish_pos,
        "text_en": report.comment,
        "source_url": source_url,
        "fetched_at": fetched_at,
    }
    _upsert(
        session,
        insert(IncidentComment).values(**comment_values),
        index_elements=["race_id", "horse_id"],
        update={k: v for k, v in comment_values.items() if k not in ("race_id", "horse_id")},
        primary_key=IncidentComment.id,
    )
    return _Written(runner=1, comment=1)


def _upsert(
    session: Session,
    statement: object,
    *,
    index_elements: list[str],
    update: Mapping[str, object],
    primary_key: object,
) -> int:
    """Run one upsert and return the row's primary key, new or existing.

    `DO UPDATE` rather than `DO NOTHING` even when the values are unchanged, because
    `DO NOTHING` returns no row and would cost a second query to find the id.
    """
    result = session.execute(
        statement.on_conflict_do_update(  # type: ignore[attr-defined]
            index_elements=index_elements, set_=dict(update)
        ).returning(primary_key)
    )
    session.flush()
    return int(result.scalar_one())


class RunRecord:
    """Handle on the open `ingest_runs` row, for outcomes other than ok/failed."""

    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        self.status: str | None = None
        self.error: str | None = None

    def pending(self, reason: str) -> None:
        """Close as `report_pending` — the page is not published *yet*.

        Distinct from a failure because a retry is the right response and an alert is
        not: HKJC posts the stewards' report hours after the results (T14).
        """
        self.status = "report_pending"
        self.error = reason


@contextmanager
def record_run(source: str, race_date: dt.date | None) -> Iterator[RunRecord]:
    """Record one ingestion attempt, whatever happens to it.

    Writes `running` immediately, so a process killed mid-meeting is visible as a run
    that never finished, then `ok`, `failed`, `fallback_detected` or `report_pending`
    on the way out. Each write is its own transaction — see the module docstring.

    Exceptions are recorded and re-raised, never swallowed: the caller decides
    whether one bad meeting stops a backfill.
    """
    started_at = dt.datetime.now(dt.UTC)
    with session_scope() as session:
        row = IngestRun(source=source, race_date=race_date, status="running", started_at=started_at)
        session.add(row)
        session.flush()
        record = RunRecord(row.id)

    try:
        yield record
    except BaseException as error:
        # A backfill over generated candidate dates expects two rejections for every
        # hit, so recording those as failures would bury the ones that need a human.
        # MeetingHeaderMissingError is deliberately *not* in this category: it means
        # the guard can no longer tell real pages from substituted ones.
        status = "fallback_detected" if isinstance(error, FallbackDetectedError) else "failed"
        _close(record.run_id, status=status, error=str(error))
        raise

    _close(record.run_id, status=record.status or "ok", error=record.error)


def _close(run_id: int, *, status: str, error: str | None) -> None:
    with session_scope() as session:
        row = session.get(IngestRun, run_id)
        if row is None:  # pragma: no cover — only reachable if someone truncates the table
            return
        row.status = status
        row.error = error
        row.finished_at = dt.datetime.now(dt.UTC)
