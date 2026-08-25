"""Filling `name_zh` for a meeting that is already ingested.

T6 built Chinese resolution and nothing ever fed it: after a full two-season
backfill, `horses.name_zh`, `jockeys.name_zh` and `trainers.name_zh` are all
zero-filled. Spec §1 Q5 asks about a horse by its Chinese name, so it cannot resolve
its own subject.

The property these tests exist for is **no new entities**. This pass meets every
horse, jockey and trainer a second time under a different name, and the one way it
can go wrong is to write 1,976 more horses beside the ones it was supposed to
annotate. So the row counts are asserted before and after, and the pair
`包裝天王` / `PACKING KING` has to come back as one `horse_id`.

The join is the horse's own link — the Chinese page carries `horseid=HK_2024_K570`
exactly as the English one does. Nothing matches on a name and nothing joins by row
position.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select, update
from tests.doubles import RecordingFetcher

from paddock.db.models import FetchedPage, Horse, IngestRun, Jockey, Meeting, Trainer, Watermark
from paddock.db.session import session_scope
from paddock.ingest import pipeline
from paddock.ingest.pipeline import ingest_meeting
from paddock.ingest.translations import CHINESE_RESULTS_PATH, translate_meeting
from paddock.ingest.watermark import INCIDENT_REPORT

pytestmark = pytest.mark.integration

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "html"

RACE_DATE = dt.date(2026, 4, 26)
RACECOURSE = "ST"
RACES_IN_CARD = 11

WINNER = "HK_2024_K570"
WINNER_EN = "PACKING KING"
WINNER_ZH = "包裝天王"
JOCKEY_EN = "Z Purton"
JOCKEY_ZH = "潘頓"
TRAINER_EN = "C S Shum"
TRAINER_ZH = "沈集成"


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _reset()
    yield
    _reset()


def _reset() -> None:
    """Delete the meeting, its archived pages, and every Chinese name it wrote.

    The names have to go as well as the meeting. A leftover `name_zh` makes "the
    second run changed nothing" true for the wrong reason.
    """
    with session_scope() as session:
        session.query(Meeting).filter(Meeting.race_date == RACE_DATE).delete(
            synchronize_session=False
        )
        session.query(IngestRun).filter(IngestRun.race_date == RACE_DATE).delete(
            synchronize_session=False
        )
        session.query(FetchedPage).filter(FetchedPage.url.in_(_urls())).delete(
            synchronize_session=False
        )
        session.query(Watermark).filter(Watermark.source == INCIDENT_REPORT).delete(
            synchronize_session=False
        )
        for model in (Horse, Jockey, Trainer):
            session.execute(update(model).values(name_zh=None))


def _urls() -> list[str]:
    """Every URL this meeting can be archived under, in both languages."""
    url_for = RecordingFetcher().url_for
    urls = [url_for(pipeline.REPORT_PATH, pipeline.report_params(RACE_DATE))]
    for race_no in range(1, RACES_IN_CARD + 1):
        params = pipeline.results_params(RACE_DATE, RACECOURSE, race_no)
        urls.append(url_for(pipeline.RESULTS_PATH, params))
        urls.append(url_for(CHINESE_RESULTS_PATH, params))
        urls.append(
            url_for(pipeline.SECTIONALS_PATH, pipeline.sectionals_params(RACE_DATE, race_no))
        )
    return urls


def _english_client() -> RecordingFetcher:
    """The meeting as T11 ingested it: a full card, results for Race 1 only."""
    client = RecordingFetcher()
    client.serve(
        pipeline.REPORT_PATH,
        pipeline.report_params(RACE_DATE),
        (FIXTURES / "report_20260426_valid.html").read_text(),
    )
    client.serve(
        pipeline.RESULTS_PATH,
        pipeline.results_params(RACE_DATE, RACECOURSE, 1),
        (FIXTURES / "results_20260426_ST_R1.html").read_text(),
    )
    client.serve(
        pipeline.SECTIONALS_PATH,
        pipeline.sectionals_params(RACE_DATE, 1),
        (FIXTURES / "sectional_20260426_R1.html").read_text(),
    )
    empty = (FIXTURES / "results_20260423_no_meeting.html").read_text()
    for race_no in range(2, RACES_IN_CARD + 1):
        client.serve(
            pipeline.RESULTS_PATH, pipeline.results_params(RACE_DATE, RACECOURSE, race_no), empty
        )
        client.serve(
            pipeline.SECTIONALS_PATH, pipeline.sectionals_params(RACE_DATE, race_no), empty
        )
    return client


def _chinese_client(*, race1_body: str | None = None) -> RecordingFetcher:
    """A client that answers the Chinese URLs. Nothing else is registered, so a pass
    that asks for a page it should have skipped fails loudly rather than quietly."""
    client = RecordingFetcher()
    client.serve(
        CHINESE_RESULTS_PATH,
        pipeline.results_params(RACE_DATE, RACECOURSE, 1),
        race1_body or (FIXTURES / "results_20260426_ST_R1_zh.html").read_text(),
    )
    return client


def _seed() -> None:
    ingest_meeting(_english_client(), RACE_DATE, RACECOURSE)


def _counts() -> dict[str, int]:
    with session_scope() as session:
        return {
            "horses": session.scalar(select(func.count()).select_from(Horse)) or 0,
            "jockeys": session.scalar(select(func.count()).select_from(Jockey)) or 0,
            "trainers": session.scalar(select(func.count()).select_from(Trainer)) or 0,
        }


def _names() -> list[tuple[str, str | None]]:
    """Every Chinese name in the three tables, keyed by identity."""
    with session_scope() as session:
        rows = [(h.horse_id, h.name_zh) for h in session.scalars(select(Horse))]
        rows += [(f"jockey:{j.id}", j.name_zh) for j in session.scalars(select(Jockey))]
        rows += [(f"trainer:{t.id}", t.name_zh) for t in session.scalars(select(Trainer))]
        return sorted(rows)


# ── The names land ──────────────────────────────────────────────────────────────


def test_the_horse_gets_its_chinese_name() -> None:
    _seed()
    translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)

    with session_scope() as session:
        horse = session.get(Horse, WINNER)
        assert horse is not None
        assert horse.name_en == WINNER_EN
        assert horse.name_zh == WINNER_ZH


def test_the_jockey_and_the_trainer_get_theirs() -> None:
    _seed()
    translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)

    with session_scope() as session:
        jockey = session.scalar(select(Jockey).where(Jockey.name_en == JOCKEY_EN))
        trainer = session.scalar(select(Trainer).where(Trainer.name_en == TRAINER_EN))
        assert jockey is not None and jockey.name_zh == JOCKEY_ZH
        assert trainer is not None and trainer.name_zh == TRAINER_ZH


def test_the_chinese_and_english_names_are_one_horse() -> None:
    """Spec §1 Q5 asks by Chinese name. It has to reach the same row."""
    _seed()
    translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)

    with session_scope() as session:
        by_english = session.scalar(select(Horse.horse_id).where(Horse.name_en == WINNER_EN))
        by_chinese = session.scalar(select(Horse.horse_id).where(Horse.name_zh == WINNER_ZH))
        assert by_english == by_chinese == WINNER


def test_every_runner_in_the_race_is_named() -> None:
    _seed()
    report = translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)

    assert report.runners == 14, "the whole field, not just the winner"


# ── And nothing else changes ────────────────────────────────────────────────────


def test_no_horse_jockey_or_trainer_is_created() -> None:
    """The failure this task is most exposed to: 1,976 more horses beside the real
    ones, each holding one language of one name."""
    _seed()
    before = _counts()

    report = translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)

    assert _counts() == before
    assert report.created == 0


def test_a_second_run_changes_nothing() -> None:
    _seed()
    translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)
    after_first = _names()

    second = translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)

    assert _names() == after_first
    assert second.created == 0


def test_the_second_run_makes_no_request() -> None:
    """`fetch_page` archives the Chinese page like every other, so re-running a
    season is free rather than another 1,800 requests."""
    _seed()
    translate_meeting(_chinese_client(), RACE_DATE, RACECOURSE)

    client = _chinese_client()
    translate_meeting(client, RACE_DATE, RACECOURSE)

    assert client.requests == []


# ── Races it must step over ─────────────────────────────────────────────────────


def test_a_race_with_no_english_results_is_never_asked_for_in_chinese() -> None:
    """Ten of this card's eleven races have no results page. Fetching their Chinese
    twin would buy ten pages that cannot be paired with anything."""
    _seed()
    client = _chinese_client()

    report = translate_meeting(client, RACE_DATE, RACECOURSE)

    assert report.races_without_english == list(range(2, RACES_IN_CARD + 1))
    assert len(client.requests) == 1, "one request, for the one race with results"


def test_a_chinese_page_for_another_race_is_refused() -> None:
    """This URL is new and no guard has been pointed at it. If HKJC substitutes a
    page the way its report endpoint does, `第 N 場` is what says so."""
    _seed()
    wrong_race = (
        (FIXTURES / "results_20260426_ST_R1_zh.html").read_text().replace("第 1 場", "第 5 場")
    )

    report = translate_meeting(_chinese_client(race1_body=wrong_race), RACE_DATE, RACECOURSE)

    assert report.race_number_mismatches == [1]
    assert report.runners == 0
    with session_scope() as session:
        horse = session.get(Horse, WINNER)
        assert horse is not None and horse.name_zh is None, "nothing was written"
