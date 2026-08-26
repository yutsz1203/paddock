"""The demo, driven headlessly.

`app/streamlit_app.py` is the widget layer, and the acceptance criteria for T22 are
all statements about what a visitor sees: the data range, the stated scope, the
language toggle, and a source card carrying the original comment. Streamlit's own
`AppTest` runs the script without a browser, so those four are checked rather than
eyeballed.

**Marked `ui` and deselected by default.** `streamlit` is an optional extra, and the
rest of the suite runs on a bare `uv sync`. CI installs every extra and runs this
file in a step of its own. Locally: `make test-ui`.

The API is a `MockTransport` seeded into session state before the first run. So no
server, no Postgres and no model — the seam is one `if "api" not in st.session_state`
in the app, which is also how the real client avoids being rebuilt on every rerun.
"""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from paddock.ui.client import ApiClient
from paddock.ui.text import strings

pytestmark = pytest.mark.ui

APP = str(pathlib.Path(__file__).parent.parent.parent / "app" / "streamlit_app.py")

COVERAGE = {
    "meetings": 176,
    "first_date": "2024-09-08",
    "last_date": "2026-07-15",
    "seasons": ["2024-25", "2025-26"],
}

TROUBLE = "Was hampered approaching the 800 Metres and lost ground."

ANSWER = (
    'event: token\ndata: {"text": "SETANTA was hampered "}\n\n'
    'event: token\ndata: {"text": "near the 800 [S1]."}\n\n'
    "event: sources\ndata: "
    + json.dumps(
        {
            "sources": [
                {
                    "marker": "S1",
                    "kind": "comment",
                    "text": TROUBLE,
                    "reference": "incident_comment:77",
                }
            ]
        }
    )
    + "\n\n"
    'event: done\ndata: {"route": "vector", "horse_id": "HK_2024_K570",'
    ' "horse_name": "SETANTA", "abstained": false, "attempts": 1}\n\n'
)

REFUSAL = (
    'event: token\ndata: {"text": "I have no evidence for that."}\n\n'
    'event: sources\ndata: {"sources": []}\n\n'
    'event: done\ndata: {"route": "vector", "abstained": true, "attempts": 0}\n\n'
)

NO_MODEL = 'event: error\ndata: {"message": "llm_not_configured"}\n\n'


def _app(*, answer: str = ANSWER, coverage: dict[str, object] | None = None) -> AppTest:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/coverage":
            return httpx.Response(200, json=coverage if coverage is not None else COVERAGE)
        return httpx.Response(200, content=answer.encode())

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api"] = ApiClient(
        httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(handle))
    )
    return at


def _page_text(at: AppTest) -> str:
    """Everything the page rendered, as one string."""
    parts = [element.value for element in at.markdown] + [element.value for element in at.caption]
    parts += [element.value for element in at.info] + [element.value for element in at.error]
    parts += [element.value for element in at.title] + [element.value for element in at.warning]
    parts += [element.label for element in at.expander]
    return "\n".join(str(part) for part in parts)


# ── The banner and the stated scope ─────────────────────────────────────────────


def test_the_data_range_is_stated_before_any_question_is_asked() -> None:
    at = _app().run()

    assert not at.exception
    banner = "\n".join(element.value for element in at.info)
    assert "2024-25 and 2025-26 seasons" in banner
    assert "15 July 2026" in banner
    assert "176 meetings" in banner


def test_the_scope_is_stated_next_to_the_data_range() -> None:
    """A visitor who meets a refusal has to be able to tell a stated limit from a
    broken feature (ADR-004). That sentence is on the page from the first paint."""
    at = _app().run()

    labels = [element.label for element in at.expander]
    assert strings("en").scope_title in labels

    scope = next(element for element in at.expander if element.label == strings("en").scope_title)
    body = "\n".join(str(child.value) for child in scope.markdown)
    assert "Happy Valley" in body
    assert "ADR-004" in body


def test_an_empty_corpus_says_so_rather_than_naming_a_range() -> None:
    at = _app(coverage={"meetings": 0, "first_date": None, "last_date": None, "seasons": []}).run()

    assert not at.exception
    assert strings("en").banner_empty in _page_text(at)


# ── The language toggle ─────────────────────────────────────────────────────────


def test_the_page_starts_in_english() -> None:
    at = _app().run()

    assert strings("en").tagline in _page_text(at)


def test_switching_the_toggle_translates_the_whole_page() -> None:
    at = _app().run()

    at.segmented_control[0].set_value("中文").run()

    text = _page_text(at)
    assert strings("zh").tagline in text
    assert strings("zh").scope_title in [element.label for element in at.expander]
    assert "2026年7月15日" in text
    assert strings("en").tagline not in text


def test_the_toggle_says_it_changes_the_interface_and_not_the_answer() -> None:
    """The model answers in the language of the question. Without this line the
    toggle reads as broken the first time someone sets 中文 and asks in English."""
    at = _app().run()

    assert strings("en").language_note in _page_text(at)


# ── Answering ───────────────────────────────────────────────────────────────────


def test_an_answer_renders_and_keeps_its_citation_markers() -> None:
    at = _app().run()

    at.chat_input[0].set_value("Did SETANTA have trouble?").run()

    assert not at.exception
    assert "[S1]" in _page_text(at)


def test_a_citation_becomes_a_card_holding_the_original_comment() -> None:
    at = _app().run()

    at.chat_input[0].set_value("Did SETANTA have trouble?").run()

    card = next(element for element in at.expander if "S1" in element.label)
    assert "incident_comment:77" in card.label
    assert TROUBLE in "\n".join(str(child.value) for child in card.markdown)


def test_the_question_and_the_answer_both_stay_on_screen() -> None:
    at = _app().run()

    at.chat_input[0].set_value("Did SETANTA have trouble?").run()

    text = _page_text(at)
    assert "Did SETANTA have trouble?" in text
    assert "SETANTA was hampered" in text


def test_a_refusal_is_labelled_as_the_intended_behaviour() -> None:
    at = _app(answer=REFUSAL).run()

    at.chat_input[0].set_value("Who is the best jockey at Happy Valley?").run()

    assert strings("en").abstained_note in _page_text(at)


def test_an_unconfigured_model_is_reported_in_the_visitors_language() -> None:
    at = _app(answer=NO_MODEL).run()

    at.chat_input[0].set_value("Did SETANTA have trouble?").run()

    assert not at.exception
    assert strings("en").error_llm in _page_text(at)


def test_an_api_that_is_not_running_is_a_message_not_a_traceback() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api"] = ApiClient(
        httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(refuse))
    )
    at.run()

    assert not at.exception
    assert any("http://api.test" in element.value for element in at.error)
