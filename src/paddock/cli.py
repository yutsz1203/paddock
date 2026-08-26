"""The command line — `uv run paddock ...`.

Deliberately thin. Every command is a few lines over a function that is tested on its
own, because a CLI that holds logic is logic that can only be exercised through
argument parsing. What lives here is what the command line genuinely owns:

**Turning strings into typed values.** `--date 20260426` becomes a `date`, `--course`
becomes one of two racecourses. A typo is rejected before anything opens a
transaction or makes a request.

**Turning a domain exception into a line someone can read.** A served fallback, a
markup change, an endpoint that moved — these are all *expected* outcomes of pointing
a scraper at a site nobody promised us. They exit non-zero with one sentence, because
the operator needs to know a date was skipped, not to read a stack trace.

**Turning a long run into something to watch.** `ingest season` is one to two hours
of network-bound work. It prints each date as it lands rather than a summary at the
end, because a silent process and a hung one look identical.

## What is not here yet

`ingest upcoming` (T13), `ingest since` and `schedule` (T14) and `eval` (T19) are in
the spec's command list and are not built, because the code underneath them is not
built either. They are absent rather than stubbed: a command that exists and does
nothing is worse than one that does not exist.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

import typer
from sqlalchemy import select

from paddock.db.models import Meeting
from paddock.db.session import session_scope
from paddock.demo.seed import (
    DEFAULT_MEETINGS,
    SeedReport,
    prune_to_recent_meetings,
    seed_report,
)
from paddock.embed.corpus import (
    MeetingOutcome,
    benchmark_search,
    chunk_coverage,
    embed_corpus,
)
from paddock.embed.embedder import get_embedder
from paddock.embed.store import embed_meeting
from paddock.ingest.backfill import Outcome, backfill
from paddock.ingest.date_guard import FallbackDetectedError, MeetingHeaderMissingError
from paddock.ingest.dates import dates_for_season, season_bounds
from paddock.ingest.http import HkjcClient
from paddock.ingest.incident_report import ReportParseError
from paddock.ingest.integrity import IntegrityReport, check_integrity, verify_season
from paddock.ingest.pipeline import ingest_meeting
from paddock.ingest.racing_calendar import published_meetings
from paddock.ingest.results import ResultsParseError
from paddock.ingest.translations import (
    SeasonTranslation,
    TranslationOutcome,
    translate_season,
    translation_coverage,
)

app = typer.Typer(
    name="paddock",
    help="Question answering over Hong Kong Jockey Club racing data.",
    no_args_is_help=True,
)
ingest_app = typer.Typer(help="Fetch and store racing data.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
check_app = typer.Typer(help="Audit what was ingested.", no_args_is_help=True)
app.add_typer(check_app, name="check")
demo_app = typer.Typer(help="Build and inspect the committed demo dataset.", no_args_is_help=True)
app.add_typer(demo_app, name="demo")


class Racecourse(StrEnum):
    """The only two in Hong Kong. An enum so a typo is a parse error, not a meeting."""

    ST = "ST"
    HV = "HV"


# The errors that mean "the site did not give us what we asked for". Each is a normal
# outcome of scraping, and each should read as one sentence — see the module docstring.
_EXPECTED = (
    FallbackDetectedError,
    MeetingHeaderMissingError,
    ReportParseError,
    ResultsParseError,
)


def _parse_date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise typer.BadParameter(f"{value!r} is not a date in YYYYMMDD form") from None


def _parse_season(value: str) -> str:
    # Validated here rather than at the first request, because the run it would
    # otherwise start is an hour long and 3,700 requests.
    try:
        season_bounds(value)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    return value


DateOption = typer.Option(
    ..., "--date", metavar="YYYYMMDD", help="Meeting date.", parser=_parse_date
)
SeasonOption = typer.Option(
    ...,
    "--season",
    metavar="YYYY-YY",
    help="Season to backfill, e.g. 2025-26.",
    parser=_parse_season,
)
CourseOption = typer.Option(..., "--course", help="Racecourse: ST (Sha Tin) or HV (Happy Valley).")

# The embed command takes a meeting or the whole corpus, so neither half of the
# meeting pair can be required. What replaces `...` is the check in `_embed_target`:
# a bare `paddock embed` is a mistake, not a request to start a half-hour run.
OptionalDateOption = typer.Option(
    None, "--date", metavar="YYYYMMDD", help="Meeting date.", parser=_parse_date
)
OptionalCourseOption = typer.Option(
    None, "--course", help="Racecourse: ST (Sha Tin) or HV (Happy Valley)."
)
AllOption = typer.Option(False, "--all", help="Embed every stored meeting instead of one.")
SinceOption = typer.Option(
    None,
    "--since",
    metavar="YYYYMMDD",
    help="With --all: skip meetings before this date.",
    parser=_parse_date,
)
UntilOption = typer.Option(
    None,
    "--until",
    metavar="YYYYMMDD",
    help="With --all: skip meetings after this date.",
    parser=_parse_date,
)


@ingest_app.command("meeting")
def ingest_meeting_command(
    date: dt.date = DateOption,
    course: Racecourse = CourseOption,
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Re-fetch every page instead of reading the archive. For a report HKJC "
        "corrected after publication — not as a cache buster.",
    ),
) -> None:
    """Ingest one meeting: incident report, results and sectionals."""
    with HkjcClient() as client:
        try:
            result = ingest_meeting(client, date, course.value, refresh=refresh)
        except _EXPECTED as error:
            typer.secho(str(error), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None

    typer.echo(
        f"{date.isoformat()} {course.value}: "
        f"{result.races} races, {result.runners} runners, {result.comments} comments"
    )
    if result.races_without_results:
        # Named rather than counted: "10 races had no results" is a number to shrug
        # at, a list of race numbers is something to go and check.
        missing = ", ".join(str(n) for n in result.races_without_results)
        typer.secho(f"no results page for races {missing}", fg=typer.colors.YELLOW, err=True)

    typer.echo(f"next: uv run paddock embed --date {date:%Y%m%d} --course {course.value}")


@app.command("embed")
def embed_command(
    date: dt.date | None = OptionalDateOption,
    course: Racecourse | None = OptionalCourseOption,
    all_meetings: bool = AllOption,
    since: dt.date | None = SinceOption,
    until: dt.date | None = UntilOption,
) -> None:
    """Embed incident comments, skipping text that has not changed.

    One meeting with `--date` and `--course`, or the whole corpus with `--all`.
    """
    meeting = _embed_target(date, course, all_meetings=all_meetings, since=since, until=until)
    if meeting is None:
        _embed_corpus(since=since, until=until)
    else:
        _embed_one_meeting(*meeting)


def _embed_target(
    date: dt.date | None,
    course: Racecourse | None,
    *,
    all_meetings: bool,
    since: dt.date | None,
    until: dt.date | None,
) -> tuple[dt.date, Racecourse] | None:
    """The meeting to embed, or `None` for the whole corpus.

    Every ambiguous combination is refused here, before the model is loaded or a walk
    is started. A bare `paddock embed` is the one that matters: read as "the corpus"
    it starts half an hour of CPU that nobody asked for.
    """
    if all_meetings:
        if date is not None or course is not None:
            raise typer.BadParameter("--all embeds every meeting; drop --date and --course")
        return None

    if since is not None or until is not None:
        raise typer.BadParameter("--since and --until narrow --all, not one meeting")
    if date is None and course is None:
        raise typer.BadParameter("give --date and --course for one meeting, or --all")
    if date is None or course is None:
        raise typer.BadParameter("--date and --course go together")
    return date, course


def _embed_one_meeting(date: dt.date, course: Racecourse) -> None:
    # Looked up before the embedder is asked for, so a mistyped date costs nothing:
    # `get_embedder` is cheap but the first `embed()` behind it reads 2.2 GB.
    with session_scope() as session:
        meeting_id = session.scalar(
            select(Meeting.id).where(Meeting.race_date == date, Meeting.racecourse == course.value)
        )

    if meeting_id is None:
        typer.secho(
            f"no meeting on {date.isoformat()} at {course.value} — ingest it first",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("embedding (bge-m3 loads on first use, ~1 minute)...")
    with session_scope() as session:
        result = embed_meeting(session, meeting_id=meeting_id, embedder=get_embedder())

    typer.echo(
        f"{result.embedded} chunks embedded of {result.total} "
        f"from {result.comments} comments ({result.unchanged} unchanged, {result.deleted} deleted)"
    )


def _embed_corpus(*, since: dt.date | None, until: dt.date | None) -> None:
    """Walk the corpus, printing each meeting as it lands.

    Half an hour of CPU. Restartable: run it again and the meetings already done are
    re-read but not re-encoded, so an interrupted run is resumed rather than repeated.
    """
    typer.echo("embedding the corpus (bge-m3 loads on first use, ~1 minute)...")
    report = embed_corpus(
        embedder=get_embedder(), since=since, until=until, on_outcome=_announce_meeting
    )

    if report.orphans_deleted:
        typer.secho(
            f"{report.orphans_deleted} orphan chunks deleted — their comment was re-ingested "
            f"under a new id",
            fg=typer.colors.YELLOW,
        )
    typer.echo(
        f"{report.meetings} meetings: {report.embedded} chunks embedded of {report.total} "
        f"from {report.comments} comments "
        f"({report.unchanged} unchanged, {report.deleted} deleted)"
    )
    for outcome in report.failed:
        typer.secho(f"  {outcome.race_date} failed: {outcome.error}", fg=typer.colors.RED, err=True)
    if report.failed:
        # Nothing was corrupted — each failed meeting rolled its own transaction back
        # — but a corpus with a hole in it must not exit green.
        raise typer.Exit(1)


def _announce_meeting(outcome: MeetingOutcome) -> None:
    """One line per meeting. A half-hour run that prints nothing looks hung."""
    result = outcome.result
    if result is None:
        typer.secho(f"  {outcome.race_date} {outcome.racecourse}  failed", fg=typer.colors.RED)
        return
    typer.secho(
        f"  {outcome.race_date} {outcome.racecourse}  "
        f"{result.embedded} embedded, {result.unchanged} unchanged",
        fg=typer.colors.GREEN if result.embedded else typer.colors.WHITE,
    )


@ingest_app.command("season")
def ingest_season_command(
    season: str = SeasonOption,
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-ingest meetings already stored, re-fetching their pages."
    ),
) -> None:
    """Backfill a whole season, then audit what came out.

    Restartable: dates already stored are skipped, and dates already checked and
    rejected are re-checked from the page archive without a request. So a run that
    stops at meeting 60 of 88 is resumed by running this again.
    """
    with HkjcClient() as client:
        dates, source = dates_for_season(client, season)
        typer.echo(f"{season}: {len(dates)} dates from the {source}")

        try:
            report = backfill(client, dates, refresh=refresh, on_outcome=_announce)
        except MeetingHeaderMissingError as error:
            # The one failure worth stopping for: past it, the guard can no longer
            # tell a real page from a substituted one. What was written stands.
            typer.secho(f"stopped: {error}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None

    typer.echo(
        f"{len(dates)} dates: {len(report.ingested)} ingested, {len(report.rejected)} rejected, "
        f"{len(report.skipped)} skipped, {len(report.failed)} failed"
    )
    typer.echo(f"{report.meetings} meetings, {report.races} races, {report.comments} comments")

    for race_date, races in report.races_without_results:
        missing = ", ".join(str(n) for n in races)
        typer.secho(f"{race_date}: no results page for races {missing}", fg=typer.colors.YELLOW)

    clean = _report_integrity(check_integrity())
    if report.failed or not clean:
        # Nothing was corrupted — each failed meeting rolled its own transaction back
        # — but a season with a hole in it must not exit green.
        raise typer.Exit(1)


def _announce(outcome: Outcome) -> None:
    """One line per date, as it happens. See the module docstring."""
    colour = {
        "ingested": typer.colors.GREEN,
        "rejected": typer.colors.BLUE,
        "skipped": typer.colors.WHITE,
        "failed": typer.colors.RED,
    }[outcome.status]
    detail = outcome.detail or ""
    if outcome.ingest is not None:
        detail = (
            f"{outcome.ingest.races} races, {outcome.ingest.runners} runners, "
            f"{outcome.ingest.comments} comments"
        )
    typer.secho(f"  {outcome.race_date}  {outcome.status:<9} {detail}", fg=colour)


@ingest_app.command("translations")
def ingest_translations_command(
    season: str = SeasonOption,
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-fetch the Chinese pages instead of reading the archive."
    ),
) -> None:
    """Fill the Chinese name of every horse, jockey and trainer in a stored season.

    T6 built the resolution and nothing ever fed it, so spec §1 Q5 — a question about
    a horse by its Chinese name — could not resolve its own subject. This annotates
    what the backfill already stored; it never adds a meeting.

    About one request per race, and re-running is free: the Chinese pages are
    archived like every other page.
    """
    with HkjcClient() as client:
        report = translate_season(client, season, refresh=refresh, on_outcome=_announce_names)

    if not report.outcomes:
        typer.secho(
            f"no meetings stored for {season} — run `paddock ingest season --season {season}`",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"{report.meetings} meetings: {report.races} races, {report.runners} runners named")
    _report_translation_coverage()
    if not _translation_findings(report):
        raise typer.Exit(1)


def _announce_names(outcome: TranslationOutcome) -> None:
    """One line per meeting. Half an hour of requests that prints nothing looks hung."""
    if outcome.result is None:
        typer.secho(
            f"  {outcome.race_date} {outcome.racecourse}  failed: {outcome.error}",
            fg=typer.colors.RED,
        )
        return
    typer.secho(
        f"  {outcome.race_date} {outcome.racecourse}  "
        f"{outcome.result.races} races, {outcome.result.runners} runners",
        fg=typer.colors.GREEN if outcome.result.runners else typer.colors.WHITE,
    )


def _report_translation_coverage() -> None:
    """How much of the corpus now has a Chinese name. `horses` is what Q5 needs."""
    with session_scope() as session:
        coverage = translation_coverage(session)

    typer.echo(
        f"named: {coverage.horses_named}/{coverage.horses} horses, "
        f"{coverage.jockeys_named}/{coverage.jockeys} jockeys, "
        f"{coverage.trainers_named}/{coverage.trainers} trainers"
    )


def _translation_findings(report: SeasonTranslation) -> bool:
    """Print everything worth a human's attention, and say whether the run was clean."""
    if report.created:
        # The one failure this pass is exposed to: a horse, jockey or trainer written
        # beside the real one, holding one language of one name. Loud, not counted in
        # passing — it means the join between the two pages broke.
        typer.secho(
            f"{report.created} horses, jockeys or trainers were CREATED by this pass — "
            f"they should all have existed already",
            fg=typer.colors.RED,
            err=True,
        )

    for race_date, races in report.race_number_mismatches:
        named = ", ".join(str(n) for n in races)
        typer.secho(
            f"{race_date}: the Chinese page named another race for races {named}",
            fg=typer.colors.RED,
            err=True,
        )

    for race_date, races in report.races_without_chinese:
        named = ", ".join(str(n) for n in races)
        typer.secho(
            f"{race_date}: no Chinese results page for races {named}", fg=typer.colors.YELLOW
        )

    for outcome in report.failed:
        typer.secho(f"  {outcome.race_date} failed: {outcome.error}", fg=typer.colors.RED, err=True)

    return not (report.failed or report.created or report.race_number_mismatches)


@check_app.command("integrity")
def check_integrity_command() -> None:
    """Re-derive every meeting's date from its archived page, and look for duplicates."""
    if not _report_integrity(check_integrity()):
        raise typer.Exit(1)


def _report_integrity(report: IntegrityReport) -> bool:
    """Print the audit and say whether it passed. Shared by both commands."""
    if report.clean:
        typer.secho(
            f"integrity: {report.meetings_checked} meetings checked, clean", fg=typer.colors.GREEN
        )
        return True

    typer.secho(
        f"integrity: {report.meetings_checked} meetings checked, {len(report.findings)} findings",
        fg=typer.colors.RED,
        err=True,
    )
    for finding in report.findings:
        dates = ", ".join(day.isoformat() for day in finding.race_dates)
        typer.secho(f"  {finding.check}  {dates}: {finding.detail}", fg=typer.colors.RED, err=True)
    return False


@check_app.command("vectors")
def check_vectors_command(
    repeats: int = typer.Option(10, help="Timed searches per probe query."),
    budget_ms: float = typer.Option(
        100.0, "--budget-ms", help="Ceiling on p95 search latency (plan T12)."
    ),
) -> None:
    """Say whether the corpus is fully embedded, and how fast it answers.

    Two questions, because either alone passes on a corpus nobody can use: an index
    that is fast because it is empty, and a complete one that takes a second a query.
    """
    with session_scope() as session:
        coverage = chunk_coverage(session)

    typer.echo(
        f"coverage: {coverage.chunks} chunks from {coverage.with_chunks} "
        f"of {coverage.comments} comments"
    )
    if coverage.without_chunks:
        typer.secho(
            f"  {coverage.without_chunks} comments have no chunk — run `paddock embed --all`",
            fg=typer.colors.RED,
            err=True,
        )
    if coverage.orphans:
        # Worse than a missing chunk: this one is in the index and answers questions,
        # citing a comment id that no longer resolves. `embed --all` removes them.
        typer.secho(
            f"  {coverage.orphans} orphan chunks cite a comment that is gone — run "
            f"`paddock embed --all`",
            fg=typer.colors.RED,
            err=True,
        )
    if not coverage.chunks:
        typer.secho("  nothing embedded — run `paddock embed --all`", fg=typer.colors.RED, err=True)
    if not coverage.complete or not coverage.chunks:
        raise typer.Exit(1)

    typer.echo("timing (bge-m3 loads on first use, ~1 minute)...")
    with session_scope() as session:
        latency = benchmark_search(session, embedder=get_embedder(), repeats=repeats)

    over = latency.p95_ms > budget_ms
    typer.secho(
        f"latency over {latency.samples} searches: p50 {latency.p50_ms:.1f} ms, "
        f"p95 {latency.p95_ms:.1f} ms, max {latency.max_ms:.1f} ms",
        fg=typer.colors.RED if over else typer.colors.GREEN,
    )
    if over:
        typer.secho(f"  p95 is over the {budget_ms:.0f} ms budget", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@check_app.command("season")
def check_season_command(season: str = SeasonOption) -> None:
    """Hold a backfilled season up against the fixture list HKJC published for it.

    Independent of everything the backfill used: the walk is driven by generated
    candidates and the date guard, neither of which has seen this calendar.
    """
    published = published_meetings(season)
    if published is None:
        typer.secho(
            f"no published calendar for {season} — add one at "
            f"src/paddock/data/racing_calendar_{season}.json",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    report = verify_season(season, published)
    # Abandoned meetings are named in the header rather than left out of it: "88
    # announced, 87 ingested" reads as a lost meeting until you know one never ran.
    abandoned = f", {report.abandoned} abandoned" if report.abandoned else ""
    typer.echo(
        f"{season}: {report.published} announced by HKJC{abandoned}, {report.ingested} ingested"
    )

    # Named rather than counted, in both directions: a count is something to worry
    # at, and a list of dates is something to go and re-run.
    for day in report.missing:
        typer.secho(f"  missing      {day} — announced, not in the corpus", fg=typer.colors.YELLOW)
    for day in report.unpublished:
        typer.secho(
            f"  unannounced  {day} — in the corpus, not on the calendar", fg=typer.colors.YELLOW
        )
    for day, announced, stored in report.venue_mismatches:
        typer.secho(
            f"  wrong venue  {day} — calendar says {announced}, we stored {stored}",
            fg=typer.colors.RED,
            err=True,
        )

    if not report.passed:
        raise typer.Exit(1)
    typer.secho(f"{season}: verified against the published calendar", fg=typer.colors.GREEN)


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, "--reload", help="Restart on code changes."),
) -> None:
    """Run the API."""
    # Imported here so `paddock ingest` does not pay for uvicorn's import.
    import uvicorn

    uvicorn.run("paddock.api.main:app", host=host, port=port, reload=reload)


# ── The demo dataset ────────────────────────────────────────────────────────────


MeetingsOption = typer.Option(
    DEFAULT_MEETINGS, "--meetings", help="How many of the most recent meetings to keep."
)
YesOption = typer.Option(
    False, "--yes", help="Confirm the deletion. Required, because nothing undoes it."
)


@demo_app.command("prune")
def demo_prune_command(
    meetings: int = MeetingsOption,
    yes: bool = YesOption,
) -> None:
    """Cut this database down to the most recent meetings, for `data/seed/`.

    Destructive and irreversible. Point it at a copy of the corpus, never at the
    corpus — `make seed` makes that copy for you.
    """
    if not yes:
        # A flag rather than a prompt: `make seed` is not interactive, and a prompt
        # that a script answers for you is a confirmation nobody gave.
        typer.secho(
            "demo prune deletes every meeting outside the slice, and the page archive "
            "with them. Nothing undoes it. Re-run with --yes, against a copy.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    with session_scope() as session:
        report = prune_to_recent_meetings(session, meetings=meetings)

    _report_seed(report)


@demo_app.command("report")
def demo_report_command() -> None:
    """State what this database holds. Reads only.

    `make demo` runs this after restoring, because a dump is an opaque binary and
    the operator deserves to be told what came out of it.
    """
    with session_scope() as session:
        report = seed_report(session)

    if not report.meetings:
        typer.secho(
            "no meetings — restore data/seed/paddock_demo.dump, or run an ingest",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    _report_seed(report)


def _report_seed(report: SeedReport) -> None:
    """One block, in the order a reader asks: how much, over what, and of what."""
    first = report.first_date.isoformat() if report.first_date else "-"
    last = report.last_date.isoformat() if report.last_date else "-"
    typer.echo(f"{report.meetings} meetings, {first} to {last}")
    typer.echo(
        f"  {report.races} races, {report.runners} runners, {report.comments} comments, "
        f"{report.chunks} chunks"
    )
    typer.echo(f"  {report.horses} horses, {report.jockeys} jockeys, {report.trainers} trainers")
    if report.pages:
        # Not an error here — the corpus is meant to have them. It is only the
        # committed slice that must not, and `make seed` is what checks that.
        typer.echo(f"  {report.pages} archived pages")
