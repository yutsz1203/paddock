"""Pure helpers for entity resolution — no database involved.

Two extraction problems that look trivial and are not:

**Horse IDs live in links, not text.** The visible cell says ``PACKING KING (K570)``,
which gives a brand number but not the import year. The anchor beside it points at
``horseid=HK_2024_K570``, which is the full identity. The parameter is spelled
``horseid`` on some pages and ``HorseID`` on others.

**Jockey names carry weight claims.** An apprentice appears as ``Y L Chung (-2)`` on
the incident report and as ``Y L Chung`` on the results page. Six of the 33 distinct
jockey strings in a single meeting carry a claim, so without normalisation the same
rider becomes two rows and every per-jockey statistic silently splits in half.
"""

from __future__ import annotations

import pytest

from paddock.ingest.entities import normalise_person_name, parse_horse_id


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("/en-us/local/information/horse?horseid=HK_2024_K570", "HK_2024_K570"),
        ("/en-us/local/information/horse?HorseID=HK_2024_K570", "HK_2024_K570"),
        ("/zh-hk/local/information/horse?horseid=HK_2025_L194", "HK_2025_L194"),
        (
            "https://racing.hkjc.com/en-us/local/information/horse?horseid=HK_2020_E436",
            "HK_2020_E436",
        ),
    ],
)
def test_horse_id_is_read_from_the_link(href: str, expected: str) -> None:
    assert parse_horse_id(href) == expected


@pytest.mark.parametrize(
    "href",
    ["/en-us/learn-racing/know-about-horses", "", "/en-us/local/info/horse-former-name"],
)
def test_non_horse_links_yield_nothing(href: str) -> None:
    assert parse_horse_id(href) is None


def test_brand_number_is_contained_in_the_horse_id() -> None:
    """The brand number in the corpus text is the tail of the id — that is the join."""
    horse_id = parse_horse_id("/en-us/local/information/horse?horseid=HK_2024_K570")

    assert horse_id is not None
    assert horse_id.endswith("K570")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Y L Chung (-2)", "Y L Chung"),
        ("P N Wong (-7)", "P N Wong"),
        ("H Y Yuen (-10)", "H Y Yuen"),
        ("E C W Wong (-3)", "E C W Wong"),
        ("Z Purton", "Z Purton"),
        ("  K Teetan  ", "K Teetan"),
    ],
)
def test_weight_claims_are_stripped(raw: str, expected: str) -> None:
    assert normalise_person_name(raw) == expected


def test_claimed_and_unclaimed_forms_normalise_together() -> None:
    """The same rider must not become two people."""
    assert normalise_person_name("Y L Chung (-2)") == normalise_person_name("Y L Chung")


def test_chinese_names_pass_through_unchanged() -> None:
    assert normalise_person_name("潘頓") == "潘頓"


# ── Weight claims as data ───────────────────────────────────────────────────────

# The claim is not part of the rider's identity, but it *is* part of the ride:
# allotted weight = carried weight + claim, and "ridden by a 10 lb claimer" is a
# signal in its own right. Stripping it from the name must not discard it.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Y L Chung (-2)", 2), ("P N Wong (-7)", 7), ("H Y Yuen (-10)", 10), ("Z Purton", 0)],
)
def test_weight_claim_is_captured_not_just_removed(raw: str, expected: int) -> None:
    from paddock.ingest.entities import parse_weight_claim

    assert parse_weight_claim(raw) == expected
