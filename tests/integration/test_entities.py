"""Entity resolution against the database.

The identity anchor is ``horse_id`` (``HK_2024_K570``), never the name. Hong Kong
horses are renamed mid-career, so a name-keyed design would split one horse into two
and silently halve its form. It would also merge two different horses that happened
to share a name across eras.

Renames are detected during ingestion rather than read from HKJC's former-name page,
which renders client-side and serves nothing to a plain fetch. When a known horse_id
turns up under a new name, the previous name is archived to ``horse_aliases`` and the
current name updated — so the alias table maintains itself from the data we already
fetch, and a question asked with a horse's old name still resolves.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import select

from paddock.db.models import Horse, HorseAlias, Jockey, Trainer
from paddock.db.session import session_scope
from paddock.ingest.entities import resolve_horse, resolve_jockey, resolve_trainer

pytestmark = pytest.mark.integration

# Ids outside HKJC's real numbering so a failed run cannot corrupt ingested data.
TEST_HORSE = "HK_2099_Z001"
TEST_HORSE_B = "HK_2099_Z002"


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    yield
    with session_scope() as session:
        for horse_id in (TEST_HORSE, TEST_HORSE_B):
            session.query(HorseAlias).filter_by(horse_id=horse_id).delete()
            session.query(Horse).filter_by(horse_id=horse_id).delete()
        session.query(Jockey).filter(Jockey.name_en.in_(["Z Testrider", "Y L Testchung"])).delete(
            synchronize_session=False
        )
        session.query(Trainer).filter(Trainer.name_en == "C S Testshum").delete(
            synchronize_session=False
        )


# ── Creation and identity ───────────────────────────────────────────────────────


def test_unknown_horse_is_created() -> None:
    """A debutant has no history — it must be created, not rejected."""
    with session_scope() as session:
        horse = resolve_horse(
            session, horse_id=TEST_HORSE, brand_no="Z001", name_en="TEST DEBUTANT"
        )

        assert horse.horse_id == TEST_HORSE
        assert horse.brand_no == "Z001"
        assert horse.name_en == "TEST DEBUTANT"


def test_resolving_twice_returns_one_row() -> None:
    with session_scope() as session:
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="TEST DEBUTANT")
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="TEST DEBUTANT")

    with session_scope() as session:
        rows = session.scalars(select(Horse).where(Horse.horse_id == TEST_HORSE)).all()
        assert len(rows) == 1


def test_same_brand_different_import_year_are_different_horses() -> None:
    """Brand numbers are reused across eras; only the full id is unique."""
    with session_scope() as session:
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="OLD TIMER")
        resolve_horse(session, horse_id=TEST_HORSE_B, brand_no="Z002", name_en="NEW ARRIVAL")

    with session_scope() as session:
        rows = session.scalars(
            select(Horse).where(Horse.horse_id.in_([TEST_HORSE, TEST_HORSE_B]))
        ).all()
        assert len(rows) == 2


# ── Bilingual ───────────────────────────────────────────────────────────────────


def test_english_and_chinese_resolve_to_one_horse() -> None:
    """The zh-hk page carries the same horse_id, so both names land on one row."""
    with session_scope() as session:
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="PACKING TEST")
        horse = resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_zh="測試天王")

        assert horse.name_en == "PACKING TEST"
        assert horse.name_zh == "測試天王"

    with session_scope() as session:
        rows = session.scalars(select(Horse).where(Horse.horse_id == TEST_HORSE)).all()
        assert len(rows) == 1


def test_chinese_name_alone_does_not_wipe_the_english_one() -> None:
    with session_scope() as session:
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="PACKING TEST")
        horse = resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001")

        assert horse.name_en == "PACKING TEST"


# ── Renames ─────────────────────────────────────────────────────────────────────


def test_rename_keeps_one_horse_and_archives_the_old_name() -> None:
    with session_scope() as session:
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="FORMER NAME")
        horse = resolve_horse(
            session,
            horse_id=TEST_HORSE,
            brand_no="Z001",
            name_en="CURRENT NAME",
            seen_on=dt.date(2026, 4, 26),
        )

        assert horse.name_en == "CURRENT NAME"

    with session_scope() as session:
        aliases = session.scalars(select(HorseAlias).where(HorseAlias.horse_id == TEST_HORSE)).all()
        assert [(a.name, a.lang) for a in aliases] == [("FORMER NAME", "en")]


def test_rename_is_not_recorded_twice() -> None:
    """Re-ingesting the same meeting must not grow the alias table."""
    with session_scope() as session:
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="FORMER NAME")
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="CURRENT NAME")
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_en="CURRENT NAME")

    with session_scope() as session:
        aliases = session.scalars(select(HorseAlias).where(HorseAlias.horse_id == TEST_HORSE)).all()
        assert len(aliases) == 1


def test_chinese_rename_is_archived_with_its_language() -> None:
    with session_scope() as session:
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_zh="舊名")
        resolve_horse(session, horse_id=TEST_HORSE, brand_no="Z001", name_zh="新名")

    with session_scope() as session:
        aliases = session.scalars(select(HorseAlias).where(HorseAlias.horse_id == TEST_HORSE)).all()
        assert [(a.name, a.lang) for a in aliases] == [("舊名", "zh")]


# ── People ──────────────────────────────────────────────────────────────────────


def test_jockey_with_and_without_claim_is_one_person() -> None:
    """'Y L Testchung (-2)' on the report and 'Y L Testchung' on the results page."""
    with session_scope() as session:
        first = resolve_jockey(session, name_en="Y L Testchung (-2)")
        second = resolve_jockey(session, name_en="Y L Testchung")

        assert first.id == second.id
        assert first.name_en == "Y L Testchung", "the claim must not be stored in the name"


def test_jockey_chinese_name_fills_in() -> None:
    with session_scope() as session:
        resolve_jockey(session, name_en="Z Testrider")
        jockey = resolve_jockey(session, name_en="Z Testrider", name_zh="測試騎師")

        assert jockey.name_zh == "測試騎師"

    with session_scope() as session:
        rows = session.scalars(select(Jockey).where(Jockey.name_en == "Z Testrider")).all()
        assert len(rows) == 1


def test_trainer_resolution_is_idempotent() -> None:
    with session_scope() as session:
        first = resolve_trainer(session, name_en="C S Testshum")
        second = resolve_trainer(session, name_en="C S Testshum")

        assert first.id == second.id
