"""The page archive — every page HKJC serves us is kept, and read back first.

## Why keep them

Decided at Checkpoint A step 6, and the only decision on that list that cannot be
retrofitted. Keeping just `source_url` means any parser fix after the backfill costs
a re-scrape of both seasons: ~3,700 requests, an hour at 1 req/s, and the risk that
the markup moved or the page has gone. Keeping the bodies costs ~10 MB gzipped and
turns the T4/T5 parsers into pure functions over stored input — so a parser bug is a
re-parse, not a re-fetch.

## Two properties the callers depend on

**A fetch commits on its own.** `fetch_page` opens its own session rather than taking
the caller's. A meeting that fails half-way rolls its own transaction back (T10's
"no partial meeting"), and the pages it fetched must not roll back with it — paying
for them twice is exactly what the archive exists to avoid.

**The archive is read before the network.** A second run over a date already fetched
makes no request at all, which is what makes T11 restartable. `refresh=True` forces
a real request, for the one case where the page legitimately changes: a stewards'
report corrected after the fact. That stores a new version alongside the old rather
than over it, because the old one is what earlier answers cited.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from paddock.db.models import FetchedPage
from paddock.db.session import session_scope


class PageFetcher(Protocol):
    """The slice of `HkjcClient` the archive needs. Lets tests supply canned pages."""

    def url_for(self, path: str, params: Mapping[str, str] | None = None) -> str: ...

    def get_text(self, path: str, params: Mapping[str, str] | None = None) -> str: ...


@dataclass(frozen=True)
class Page:
    url: str
    body: str
    fetched_at: dt.datetime
    from_archive: bool
    """False when this call made the request. Ingestion records it so a run's true
    network cost is visible, rather than inferred from how long it took."""


def fetch_page(
    client: PageFetcher,
    path: str,
    params: Mapping[str, str] | None = None,
    *,
    refresh: bool = False,
) -> Page:
    """Return the page at `path`, from the archive when we already have it.

    Args:
        refresh: fetch even if archived, storing the result as a new version. For
            reports HKJC corrects after publication — never as a cache buster.
    """
    url = client.url_for(path, params)

    if not refresh:
        with session_scope() as session:
            archived = latest_page(session, url)
        if archived is not None:
            return archived

    body = client.get_text(path, params)
    fetched_at = dt.datetime.now(dt.UTC)

    # Its own transaction, so the page survives a caller that later fails.
    with session_scope() as session:
        store_page(session, url=url, body=body, fetched_at=fetched_at)

    return Page(url=url, body=body, fetched_at=fetched_at, from_archive=False)


def store_page(
    session: Session, *, url: str, body: str, fetched_at: dt.datetime | None = None
) -> FetchedPage:
    """Archive one page body. Versions accumulate; nothing is ever overwritten."""
    row = FetchedPage(
        url=url,
        fetched_at=fetched_at or dt.datetime.now(dt.UTC),
        body_gz=gzip.compress(body.encode(), compresslevel=6),
        sha256=hashlib.sha256(body.encode()).hexdigest(),
    )
    session.add(row)
    session.flush()
    return row


def latest_page(session: Session, url: str) -> Page | None:
    """The newest archived version of `url`, or None if we have never fetched it."""
    row = session.scalar(
        select(FetchedPage)
        .where(FetchedPage.url == url)
        .order_by(FetchedPage.fetched_at.desc(), FetchedPage.id.desc())
        .limit(1)
    )
    if row is None:
        return None

    return Page(
        url=row.url,
        body=gzip.decompress(row.body_gz).decode(),
        fetched_at=row.fetched_at,
        from_archive=True,
    )
