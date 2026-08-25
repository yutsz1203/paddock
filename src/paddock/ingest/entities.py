"""Resolving horses, jockeys and trainers to stable identities.

## The anchor is the id, never the name

Hong Kong horses are renamed mid-career. Keying identity on a name would split one
horse into two — halving its apparent form — and could merge two different horses
that shared a name across eras. HKJC's own identifier, ``HK_2024_K570``, is stable
for the life of the horse and is what every page links to.

The brand number (``K570``) is the tail of that id, and it is what appears in the
incident-report text as ``PACKING KING (K570)``. So the corpus joins to horses
without any fuzzy name matching: read the id from the link, read the brand from the
text, and they agree by construction.

## Renames maintain themselves

HKJC publishes a former-names page, but it renders client-side and serves nothing to
a plain fetch. Instead, a rename is detected during ingestion: when a known horse_id
appears under a different name, the previous name is archived to ``horse_aliases``
and the current one updated. The alias table is therefore built from data we already
fetch, and a question asked using a horse's old name still resolves.

## Weight claims are not part of a name

An apprentice rides as ``Y L Chung (-2)`` on the incident report and ``Y L Chung`` on
the results page — the number is the weight allowance for that ride, not part of who
they are. Six of the 33 distinct jockey strings in one meeting carry a claim, so
without stripping it the same rider becomes two rows and every per-jockey statistic
splits between them.

## A Chinese-only sighting parks a placeholder, and the placeholder is repaired

``jockeys.name_en`` and ``trainers.name_en`` are NOT NULL, so a person first seen on
a Chinese page is stored under their Chinese name in both columns. When the English
name arrives, it replaces that placeholder. Otherwise the row is unreachable by the
name every English page uses, and the next English-only sighting creates a second
person for one rider. A row that already carries a real English name keeps it.
"""

from __future__ import annotations

import datetime as dt
import re
from urllib.parse import parse_qs, urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from paddock.db.models import Horse, HorseAlias, Jockey, Trainer

_HORSE_ID = re.compile(r"^HK_\d{4}_[A-Z]\d{3}$")

# "(-2)", "(-10)" — the apprentice's weight allowance for that ride.
_WEIGHT_CLAIM = re.compile(r"\s*\(-(\d+)\)\s*$")


def parse_horse_id(href: str) -> str | None:
    """Extract ``HK_2024_K570`` from a horse link, or None if there isn't one.

    The query parameter is spelled ``horseid`` on some pages and ``HorseID`` on
    others, so the lookup is case-insensitive.
    """
    if not href:
        return None

    query = parse_qs(urlparse(href).query)
    for key, values in query.items():
        if key.lower() == "horseid" and values:
            candidate = values[0].strip()
            return candidate if _HORSE_ID.match(candidate) else None
    return None


def normalise_person_name(name: str) -> str:
    """Strip an apprentice's weight claim and surrounding whitespace."""
    return _WEIGHT_CLAIM.sub("", name.strip()).strip()


def parse_weight_claim(name: str) -> int:
    """Return the apprentice allowance in pounds, or 0 for a senior rider.

    Carried weight on the results page is already net of this, so the claim is the
    only route back to the weight the handicapper allotted:

        allotted = carried + claim

    It is also a feature in its own right — a 10 lb claimer and a champion jockey at
    the same weight are not the same proposition — so it is stored on the runner
    rather than discarded with the name.
    """
    match = _WEIGHT_CLAIM.search(name.strip())
    if match is None:
        return 0
    return int(match.group(1))


def resolve_horse(
    session: Session,
    *,
    horse_id: str,
    brand_no: str,
    name_en: str | None = None,
    name_zh: str | None = None,
    seen_on: dt.date | None = None,
) -> Horse:
    """Return the horse for `horse_id`, creating or updating it as needed.

    A name that differs from the stored one is treated as a rename: the old name is
    archived to `horse_aliases` and the current name replaced. Passing no name at all
    leaves existing names untouched — some pages give only one language.
    """
    horse = session.get(Horse, horse_id)

    if horse is None:
        horse = Horse(horse_id=horse_id, brand_no=brand_no, name_en=name_en, name_zh=name_zh)
        session.add(horse)
        session.flush()
        return horse

    _apply_name(session, horse, lang="en", new_name=name_en, seen_on=seen_on)
    _apply_name(session, horse, lang="zh", new_name=name_zh, seen_on=seen_on)
    session.flush()
    return horse


def _apply_name(
    session: Session,
    horse: Horse,
    *,
    lang: str,
    new_name: str | None,
    seen_on: dt.date | None,
) -> None:
    """Set a name, archiving the previous one when it genuinely changed."""
    if not new_name:
        return

    field = "name_en" if lang == "en" else "name_zh"
    current = getattr(horse, field)

    if current == new_name:
        return

    if current:
        _archive_alias(session, horse_id=horse.horse_id, name=current, lang=lang, seen_on=seen_on)

    setattr(horse, field, new_name)


def _archive_alias(
    session: Session, *, horse_id: str, name: str, lang: str, seen_on: dt.date | None
) -> None:
    """Record a former name once. Re-ingesting a meeting must not grow this table."""
    existing = session.scalar(
        select(HorseAlias).where(
            HorseAlias.horse_id == horse_id,
            HorseAlias.name == name,
            HorseAlias.lang == lang,
        )
    )
    if existing is not None:
        return

    session.add(HorseAlias(horse_id=horse_id, name=name, lang=lang, valid_to=seen_on))


def resolve_jockey(
    session: Session, *, name_en: str | None = None, name_zh: str | None = None
) -> Jockey:
    """Return the jockey, creating them if unseen. Weight claims are stripped first."""
    return _resolve_person(session, Jockey, name_en=name_en, name_zh=name_zh)


def resolve_trainer(
    session: Session, *, name_en: str | None = None, name_zh: str | None = None
) -> Trainer:
    """Return the trainer, creating them if unseen."""
    return _resolve_person(session, Trainer, name_en=name_en, name_zh=name_zh)


def _resolve_person[T: (Jockey, Trainer)](
    session: Session,
    model: type[T],
    *,
    name_en: str | None,
    name_zh: str | None,
) -> T:
    english = normalise_person_name(name_en) if name_en else None
    chinese = normalise_person_name(name_zh) if name_zh else None

    if not english and not chinese:
        raise ValueError("a jockey or trainer needs at least one name")

    person: T | None = None
    if english:
        person = session.scalar(select(model).where(model.name_en == english))
    if person is None and chinese:
        person = session.scalar(select(model).where(model.name_zh == chinese))

    if person is None:
        # name_en is NOT NULL, so a Chinese-only sighting is stored under that name
        # until an English page fills it in.
        person = model(name_en=english or chinese, name_zh=chinese)
        session.add(person)
        session.flush()
        return person

    if chinese and not person.name_zh:
        person.name_zh = chinese
    if english and person.name_en == person.name_zh:
        # The Chinese placeholder below, now that the English name has arrived. Left
        # in place the row is unreachable by the name every English page uses, so the
        # next English-only sighting would create a second person for one rider.
        person.name_en = english
    session.flush()
    return person
