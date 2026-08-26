"""Every string the demo shows, in English and Chinese.

## Why the two tables are one dataclass

A dict of keys drifts: a string added for English is missing in Chinese, nothing
fails, and a visitor who switches to 中文 meets three English labels and concludes
the bilingual claim is decoration. A frozen dataclass makes the two tables the same
shape by construction, and mypy reports a field added to one and not the other
before the demo does.

## The toggle is the interface, not the answer

The model is told to answer in the language it was asked in (`agent.prompts`), which
is the honest behaviour: an English question about a Chinese horse name should not
come back in Chinese because a switch was left on. So the toggle changes labels, and
`language_note` says so — otherwise it reads as broken the first time someone sets
中文 and asks in English.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

Language = Literal["en", "zh"]

LANGUAGES: dict[Language, str] = {"en": "English", "zh": "中文"}
"""Code to the label its own speakers read. Ordered, because it is a toggle."""

# Written out rather than taken from `strftime('%B')`, which follows the process
# locale and would render the banner in whatever language the host is configured in.
_MONTHS_EN = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class Strings:
    """One language's table."""

    title: str
    tagline: str
    language_note: str

    banner: str
    """Template. `{seasons} {season_word} {last} {meetings} {meeting_word}`."""
    banner_empty: str
    season_one: str
    season_many: str
    meeting_one: str
    meeting_many: str

    scope_title: str
    scope_body: str

    input_placeholder: str
    thinking: str

    sources_title: str
    kind_comment: str
    kind_run: str

    route_title: str
    route_sql: str
    route_vector: str
    route_both: str

    abstained_note: str
    error_llm: str
    error_generic: str
    """Template. `{name}`."""

    examples_title: str
    examples: tuple[str, ...]


_EN = Strings(
    title="paddock",
    tagline="Hong Kong racing form, answered with citations.",
    language_note="Interface language. Answers follow the language of your question.",
    banner="Data: {seasons} {season_word}, through {last} — {meetings} {meeting_word}.",
    banner_empty="No meetings are loaded. Every question will be refused.",
    season_one="season",
    season_many="seasons",
    meeting_one="meeting",
    meeting_many="meetings",
    scope_title="What this demo cannot do yet",
    scope_body=(
        "Each question must name one horse. The router reads keywords only. "
        "A question about a jockey, or about a whole race, is refused rather than "
        'answered. For example, it refuses "Who is the best jockey at Happy Valley '
        'this season?". That is a stated limit (ADR-004), not a broken feature. '
        "The queries that answer it are built next.\n\n"
        "Each question is answered on its own. The system keeps no memory of the "
        "question before it. A follow-up must name the horse again."
    ),
    input_placeholder="Ask about a horse — by name, in English or Chinese.",
    thinking="Reading the evidence…",
    sources_title="Sources",
    kind_comment="Stewards' comment",
    kind_run="Form line",
    route_title="Route",
    route_sql="SQL only",
    route_vector="Retrieval only",
    route_both="SQL and retrieval",
    abstained_note=(
        "The system found no evidence it could cite, so it refused. "
        "That is the intended behaviour, not a failure."
    ),
    error_llm="No language model is configured. Set an API key and restart the API.",
    error_generic="The request failed: {name}.",
    examples_title="Try one of these",
    examples=(
        "How did SETANTA perform in its last start?",
        "How has SETANTA gone over 1200m at Sha Tin in its last 5 runs?",
        "Did SETANTA have any trouble in running last time?",
    ),
)

_ZH = Strings(
    title="paddock",
    tagline="香港賽馬往績問答，每項事實均附出處。",
    language_note="此設定只改介面語言。回答會跟隨你提問時所用的語言。",
    banner="資料：{seasons} {season_word}，截至 {last} — 共 {meetings} {meeting_word}。",
    banner_empty="資料庫沒有任何賽馬日。所有問題都會被拒絕。",
    season_one="馬季",
    season_many="馬季",
    meeting_one="個賽馬日",
    meeting_many="個賽馬日",
    scope_title="此示範暫時做不到的事",
    scope_body=(
        "每個問題必須指明一匹馬。路由器目前只讀關鍵詞。"
        "關於騎師或整場賽事的問題會被拒絕，而不是作答，"
        "例如「今季跑馬地最佳騎師是誰？」。"
        "這是已列明的限制（ADR-004），不是故障。相關查詢會在下一階段加入。\n\n"
        "每個問題都會獨立作答。系統不會記住上一個問題。追問時必須再次寫出馬名。"
    ),
    input_placeholder="輸入馬名提問，中英文皆可。",
    thinking="正在翻查證據…",
    sources_title="出處",
    kind_comment="賽事報告評語",
    kind_run="往績紀錄",
    route_title="檢索路徑",
    route_sql="只用 SQL 查詢",
    route_vector="只用語意檢索",
    route_both="SQL 查詢加語意檢索",
    abstained_note="系統找不到可引用的證據，因此拒絕作答。這是預期行為，不是失敗。",
    error_llm="未設定語言模型。請設定 API 金鑰後重新啟動 API。",
    error_generic="請求失敗：{name}。",
    examples_title="試試這些問題",
    examples=(
        "江南盛最近狀態如何？",
        "江南盛在沙田 1200 米近五仗表現如何？",
        "江南盛上仗有沒有受阻？",
    ),
)

_TABLES: dict[Language, Strings] = {"en": _EN, "zh": _ZH}


def strings(language: Language) -> Strings:
    """The table for `language`."""
    return _TABLES[language]


def data_banner(
    language: Language,
    *,
    seasons: list[str],
    last_date: dt.date | None,
    meetings: int,
) -> str:
    """State the range the demo can answer over.

    Args:
        language: which table to render from.
        seasons: seasons that have at least one meeting, oldest first.
        last_date: the newest meeting, or None if there are none.
        meetings: how many meetings are loaded.

    Returns:
        One sentence. An empty corpus returns `banner_empty` rather than a range
        built from today's date — a banner that names a range it does not hold is
        the failure the banner exists to prevent.
    """
    table = strings(language)
    if not seasons or last_date is None:
        return table.banner_empty

    return table.banner.format(
        seasons=_join_seasons(language, seasons),
        season_word=table.season_one if len(seasons) == 1 else table.season_many,
        last=format_date(language, last_date),
        meetings=meetings,
        meeting_word=table.meeting_one if meetings == 1 else table.meeting_many,
    )


def format_date(language: Language, day: dt.date) -> str:
    """`15 July 2026`, or `2026年7月15日`."""
    if language == "zh":
        return f"{day.year}年{day.month}月{day.day}日"
    return f"{day.day} {_MONTHS_EN[day.month - 1]} {day.year}"


def _join_seasons(language: Language, seasons: list[str]) -> str:
    """`2024-25 and 2025-26`, or `2024-25 及 2025-26`."""
    if len(seasons) == 1:
        return seasons[0]
    separator, final = ("、", " 及 ") if language == "zh" else (", ", " and ")
    return separator.join(seasons[:-1]) + final + seasons[-1]


def route_name(language: Language, route: str) -> str:
    """A label for the route the API reported.

    An unknown route returns its own name rather than an empty string: T16 widens
    the router, and a new route must show up as itself instead of blanking the line
    that says how the answer was reached.
    """
    table = strings(language)
    labels = {"sql": table.route_sql, "vector": table.route_vector, "both": table.route_both}
    return labels.get(route, route)


def source_kind_name(language: Language, kind: str) -> str:
    """`comment` and `run` are what the API sends; anything else shows as itself."""
    table = strings(language)
    return {"comment": table.kind_comment, "run": table.kind_run}.get(kind, kind)
