"""The page archive: every page HKJC serves us is kept.

Decided at Checkpoint A step 6. Without it, any parser fix after the backfill costs
a re-scrape of both seasons — ~3,700 requests, an hour at 1 req/s, and the risk that
the markup moved or the page is gone. It cannot be retrofitted, because a page not
kept is not recoverable.

Two properties matter beyond "the bytes come back":

**A fetch is committed on its own.** A meeting that fails half-way must still leave
the pages it fetched behind, or a retry pays for them twice. So the archive writes in
its own transaction, not the caller's.

**The archive is consulted before the network.** That is what makes T11 restartable
and what turns the T4/T5 parsers into pure functions over stored input — a parser bug
becomes a re-parse, not a re-fetch.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import select
from tests.doubles import RecordingFetcher

from paddock.db.models import FetchedPage
from paddock.db.session import session_scope
from paddock.ingest.pages import fetch_page, latest_page

pytestmark = pytest.mark.integration

# A path no HKJC page uses, so a failed run cannot collide with archived real pages.
TEST_PATH = "/test-only/pages"


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    yield
    with session_scope() as session:
        session.query(FetchedPage).filter(FetchedPage.url.like("%/test-only/%")).delete(
            synchronize_session=False
        )


def test_fetching_stores_the_body_and_returns_it() -> None:
    client = RecordingFetcher()
    client.serve(TEST_PATH, None, "<html>first</html>")

    page = fetch_page(client, TEST_PATH)

    assert page.body == "<html>first</html>"
    assert page.from_archive is False
    with session_scope() as session:
        stored = latest_page(session, page.url)
        assert stored is not None
        assert stored.body == "<html>first</html>", "the archived copy must decompress identically"


def test_archived_bodies_are_compressed_on_disk() -> None:
    """~50 MB raw for two seasons, ~10 MB gzipped. Repeated markup compresses well."""
    client = RecordingFetcher()
    client.serve(TEST_PATH, None, "<html>" + "<td>x</td>" * 5000 + "</html>")

    page = fetch_page(client, TEST_PATH)

    with session_scope() as session:
        row = session.scalar(select(FetchedPage).where(FetchedPage.url == page.url))
        assert row is not None
        assert len(row.body_gz) < len(page.body.encode()) / 10


def test_a_second_fetch_serves_the_archive_without_a_request() -> None:
    """This is what makes a re-run of T11 free rather than another 3,700 requests."""
    client = RecordingFetcher()
    client.serve(TEST_PATH, None, "<html>first</html>")

    fetch_page(client, TEST_PATH)
    again = fetch_page(client, TEST_PATH)

    assert again.body == "<html>first</html>"
    assert again.from_archive is True
    assert client.requests == [TEST_PATH], "the second call must not hit the network"


def test_params_are_part_of_the_key() -> None:
    """`?date=…` is the whole difference between one meeting's page and another's."""
    client = RecordingFetcher()
    client.serve(TEST_PATH, {"date": "2026/04/26"}, "<html>april</html>")
    client.serve(TEST_PATH, {"date": "2025/03/12"}, "<html>march</html>")

    april = fetch_page(client, TEST_PATH, {"date": "2026/04/26"})
    march = fetch_page(client, TEST_PATH, {"date": "2025/03/12"})

    assert april.body == "<html>april</html>"
    assert march.body == "<html>march</html>"
    assert april.url != march.url


def test_refresh_keeps_the_old_version_alongside_the_new() -> None:
    """A late-corrected report must not silently overwrite what we answered from."""
    client = RecordingFetcher()
    client.serve(TEST_PATH, None, "<html>first</html>")
    fetch_page(client, TEST_PATH)

    client.serve(TEST_PATH, None, "<html>corrected</html>")
    refreshed = fetch_page(client, TEST_PATH, refresh=True)

    assert refreshed.body == "<html>corrected</html>"
    with session_scope() as session:
        rows = session.scalars(
            select(FetchedPage)
            .where(FetchedPage.url == refreshed.url)
            .order_by(FetchedPage.fetched_at)
        ).all()
        assert [r.sha256 for r in rows] != [rows[0].sha256] * 2, "two distinct versions"
        assert len(rows) == 2
        assert latest_page(session, refreshed.url).body == "<html>corrected</html>"  # type: ignore[union-attr]


def test_a_page_survives_the_callers_rollback() -> None:
    """The fetch is committed on its own, so a failed meeting does not discard it."""
    client = RecordingFetcher()
    client.serve(TEST_PATH, None, "<html>kept</html>")

    with pytest.raises(RuntimeError, match="meeting blew up"), session_scope() as session:
        page = fetch_page(client, TEST_PATH)
        session.add(FetchedPage(url="/test-only/unrelated", body_gz=b"", sha256="x" * 64))
        raise RuntimeError("meeting blew up")

    with session_scope() as session:
        assert latest_page(session, page.url) is not None, "the fetched page must survive"
        assert latest_page(session, "/test-only/unrelated") is None, "the caller's write must not"
