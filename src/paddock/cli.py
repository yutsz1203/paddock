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

## What is not here yet

`ingest season` (T11), `ingest upcoming` (T13), `ingest since` and `schedule` (T14)
and `eval` (T19) are in the spec's command list and are not built, because the code
underneath them is not built either. They are absent rather than stubbed: a command
that exists and does nothing is worse than one that does not exist.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

import typer
from sqlalchemy import select

from paddock.db.models import Meeting
from paddock.db.session import session_scope
from paddock.embed.embedder import get_embedder
from paddock.embed.store import embed_meeting
from paddock.ingest.date_guard import FallbackDetectedError, MeetingHeaderMissingError
from paddock.ingest.http import HkjcClient
from paddock.ingest.incident_report import ReportParseError
from paddock.ingest.pipeline import ingest_meeting
from paddock.ingest.results import ResultsParseError

app = typer.Typer(
    name="paddock",
    help="Question answering over Hong Kong Jockey Club racing data.",
    no_args_is_help=True,
)
ingest_app = typer.Typer(help="Fetch and store racing data.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


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


DateOption = typer.Option(
    ..., "--date", metavar="YYYYMMDD", help="Meeting date.", parser=_parse_date
)
CourseOption = typer.Option(..., "--course", help="Racecourse: ST (Sha Tin) or HV (Happy Valley).")


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
    date: dt.date = DateOption,
    course: Racecourse = CourseOption,
) -> None:
    """Embed one meeting's incident comments, skipping text that has not changed."""
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
