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
RACES_IN_CARD = 11

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _delete_everything()
    yield
    _delete_everything()


def _urls() -> list[str]:
    """Every URL either meeting could be archived under, as production builds them."""
    with HkjcClient() as client:
        urls = []
        for day in (RACE_DATE, FALLBACK_DATE):
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
        for day in (RACE_DATE, FALLBACK_DATE):
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
