"""The command line: the only way to actually run any of this.

Thin by intent. Every command is a few lines over a function that is tested
elsewhere, so what these tests are about is the part the CLI genuinely owns —
turning arguments into typed values, and turning a domain exception into a line
someone can read rather than a traceback.

No network: the fixture pages are archived under the URLs `HkjcClient` would build,
so the real client is constructed and never makes a request. That is the archive
doing its job through the production path.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Iterator

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from paddock.cli import app
from paddock.db.models import FetchedPage, IngestRun, Meeting, Watermark
from paddock.db.session import session_scope
from paddock.ingest import pipeline
from paddock.ingest.http import HkjcClient
from paddock.ingest.pages import store_page
from paddock.ingest.watermark import INCIDENT_REPORT

pytestmark = pytest.mark.integration

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "html"

RACE_DATE = dt.date(2026, 4, 26)
FALLBACK_DATE = dt.date(2026, 4, 23)
UNPARSEABLE_DATE = dt.date(2026, 4, 19)  # a real Sunday whose page will not parse
ALL_DATES = (RACE_DATE, FALLBACK_DATE, UNPARSEABLE_DATE)
RACES_IN_CARD = 11

runner = CliRunner()


def _serve_dates(monkeypatch: pytest.MonkeyPatch, dates: list[dt.date], source: str) -> None:
    """Stand in for date discovery, which is the one step the archive cannot cover.

    Every page a backfill reads comes from `fetch_page` and so from Postgres, but the
    season index is a view of today rather than a page anything cites — archiving it
    would mean reading a stale season list on the next run. So it is the one seam
    these tests replace, and it has its own unit tests in `test_dates.py`.
    """
    monkeypatch.setattr("paddock.cli.dates_for_season", lambda client, season: (dates, source))


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_everything()
    yield
    _delete_everything()


def _urls() -> list[str]:
    """Every URL either meeting could be archived under, as production builds them."""
    with HkjcClient() as client:
        urls = []
        for day in ALL_DATES:
            urls.append(client.url_for(pipeline.REPORT_PATH, pipeline.report_params(day)))
            for race_no in range(1, RACES_IN_CARD + 1):
                urls.append(
                    client.url_for(
                        pipeline.RESULTS_PATH, pipeline.results_params(day, "ST", race_no)
                    )
                )
                urls.append(
                    client.url_for(
                        pipeline.SECTIONALS_PATH, pipeline.sectionals_params(day, race_no)
                    )
                )
        return urls


def _delete_everything() -> None:
    with session_scope() as session:
        for day in ALL_DATES:
            session.query(Meeting).filter(Meeting.race_date == day).delete(
                synchronize_session=False
            )
            session.query(IngestRun).filter(IngestRun.race_date == day).delete(
                synchronize_session=False
            )
        session.query(FetchedPage).filter(FetchedPage.url.in_(_urls())).delete(
            synchronize_session=False
        )
        session.query(Watermark).filter(Watermark.source == INCIDENT_REPORT).delete(
            synchronize_session=False
        )


def _archive(day: dt.date, report_fixture: str) -> None:
    """Pre-load the archive so the real client has nothing left to fetch."""
    empty = (FIXTURES / "results_20260423_no_meeting.html").read_text()
    with HkjcClient() as client, session_scope() as session:

        def keep(path: str, params: dict[str, str], body: str) -> None:
            store_page(session, url=client.url_for(path, params), body=body)

        keep(
            pipeline.REPORT_PATH,
            pipeline.report_params(day),
            (FIXTURES / report_fixture).read_text(),
        )
        keep(
            pipeline.RESULTS_PATH,
            pipeline.results_params(day, "ST", 1),
            (FIXTURES / "results_20260426_ST_R1.html").read_text(),
        )
        keep(
            pipeline.SECTIONALS_PATH,
            pipeline.sectionals_params(day, 1),
            (FIXTURES / "sectional_20260426_R1.html").read_text(),
        )
        for race_no in range(2, RACES_IN_CARD + 1):
            keep(pipeline.RESULTS_PATH, pipeline.results_params(day, "ST", race_no), empty)
            keep(pipeline.SECTIONALS_PATH, pipeline.sectionals_params(day, race_no), empty)


# ── Arguments ───────────────────────────────────────────────────────────────────


def test_a_malformed_date_is_rejected_with_the_format_it_wanted() -> None:
    result = runner.invoke(app, ["ingest", "meeting", "--date", "26/04/2026", "--course", "ST"])

    assert result.exit_code != 0
    assert "YYYYMMDD" in result.output


def test_a_date_that_is_not_a_date_is_rejected() -> None:
    result = runner.invoke(app, ["ingest", "meeting", "--date", "20260231", "--course", "ST"])

    assert result.exit_code != 0
    assert "20260231" in result.output


def test_an_unknown_racecourse_is_rejected() -> None:
    """ST and HV are the only two. A typo must not become a meeting nobody can find."""
    result = runner.invoke(app, ["ingest", "meeting", "--date", "20260426", "--course", "XX"])

    assert result.exit_code != 0
    assert "XX" in result.output


# ── Ingesting ───────────────────────────────────────────────────────────────────


def test_ingesting_a_meeting_reports_what_it_wrote() -> None:
    _archive(RACE_DATE, "report_20260426_valid.html")

    result = runner.invoke(app, ["ingest", "meeting", "--date", "20260426", "--course", "ST"])

    assert result.exit_code == 0, result.output
    assert "11 races" in result.output
    assert "142 runners" in result.output
    assert "114 comments" in result.output
    with session_scope() as session:
        assert session.scalar(select(Meeting).where(Meeting.race_date == RACE_DATE)) is not None


def test_races_without_results_are_named_not_buried() -> None:
    """Silence here is how a season of half-ingested meetings would go unnoticed."""
    _archive(RACE_DATE, "report_20260426_valid.html")

    result = runner.invoke(app, ["ingest", "meeting", "--date", "20260426", "--course", "ST"])

    assert "no results page" in result.output
    assert "2, 3, 4, 5, 6, 7, 8, 9, 10, 11" in result.output


def test_a_served_fallback_is_one_readable_line_not_a_traceback() -> None:
    """The operator needs to know the date was skipped, not to read a stack."""
    _archive(FALLBACK_DATE, "report_20260423_fallback.html")

    result = runner.invoke(app, ["ingest", "meeting", "--date", "20260423", "--course", "ST"])

    assert result.exit_code == 1
    assert "2026-04-23" in result.output
    assert "Traceback" not in result.output


# ── Embedding ───────────────────────────────────────────────────────────────────


def test_embedding_an_unknown_meeting_says_so() -> None:
    """Before loading 2.2 GB of model, and without a traceback."""
    result = runner.invoke(app, ["embed", "--date", "20260426", "--course", "ST"])

    assert result.exit_code == 1
    assert "2026-04-26" in result.output
    assert "Traceback" not in result.output


# ── Backfilling a season ────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["2024-2025", "2024-26", "next"])
def test_a_malformed_season_is_rejected_before_a_single_request(bad: str) -> None:
    """The run it would otherwise start is an hour long and 3,700 requests."""
    result = runner.invoke(app, ["ingest", "season", "--season", bad])

    assert result.exit_code != 0
    assert "2025-26" in result.output, "the form it wanted, shown by example"


def test_a_season_names_every_meeting_as_it_lands(monkeypatch: pytest.MonkeyPatch) -> None:
    """One to two hours of runtime. A summary at the end of it and nothing before is
    indistinguishable from a hang."""
    _archive(RACE_DATE, "report_20260426_valid.html")
    _serve_dates(monkeypatch, [RACE_DATE], "index")

    result = runner.invoke(app, ["ingest", "season", "--season", "2025-26"])

    assert result.exit_code == 0, result.output
    assert "88 dates" not in result.output
    assert "2026-04-26" in result.output
    assert "ingested" in result.output
    assert "1 ingested" in result.output


def test_where_the_dates_came_from_is_stated(monkeypatch: pytest.MonkeyPatch) -> None:
    """88 dates from HKJC's own index and 140 generated guesses are different runs,
    and only one of them is expected to produce a pile of rejections."""
    _serve_dates(monkeypatch, [FALLBACK_DATE], "candidates")
    _archive(FALLBACK_DATE, "report_20260423_fallback.html")

    result = runner.invoke(app, ["ingest", "season", "--season", "2024-25"])

    assert "candidates" in result.output


def test_a_rejected_date_is_named_not_just_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rejection list is what a human reviews after a prior-season backfill."""
    _archive(FALLBACK_DATE, "report_20260423_fallback.html")
    _serve_dates(monkeypatch, [FALLBACK_DATE], "candidates")

    result = runner.invoke(app, ["ingest", "season", "--season", "2024-25"])

    assert result.exit_code == 0, "two candidates in three are not meetings — not a failure"
    assert "2026-04-23" in result.output
    assert "rejected" in result.output


def test_a_backfill_that_lost_a_meeting_exits_non_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing was corrupted — the meeting rolled back — but a season with a hole in
    it must not end in a green exit code that says otherwise."""
    _serve_dates(monkeypatch, [UNPARSEABLE_DATE], "index")
    with HkjcClient() as client, session_scope() as session:
        store_page(
            session,
            url=client.url_for(pipeline.REPORT_PATH, pipeline.report_params(UNPARSEABLE_DATE)),
            body="<html><body>Race Meeting: 19/04/2026 (Sun)</body></html>",
        )

    result = runner.invoke(app, ["ingest", "season", "--season", "2025-26"])

    assert result.exit_code == 1
    assert "failed" in result.output


# ── Auditing what came out ──────────────────────────────────────────────────────


def test_the_backfill_audits_itself_when_it_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The acceptance criterion is a post-backfill integrity query, so running it is
    not left to whoever remembers."""
    _archive(RACE_DATE, "report_20260426_valid.html")
    _serve_dates(monkeypatch, [RACE_DATE], "index")

    result = runner.invoke(app, ["ingest", "season", "--season", "2025-26"])

    assert "integrity" in result.output
    assert "1 meeting" in result.output


def test_integrity_can_be_re_run_on_its_own() -> None:
    result = runner.invoke(app, ["check", "integrity"])

    assert result.exit_code == 0, result.output
    assert "clean" in result.output


def test_integrity_says_when_there_is_nothing_to_check() -> None:
    """An empty database passing every check is not the same as a corpus that did."""
    result = runner.invoke(app, ["check", "integrity"])

    assert "0 meetings" in result.output
