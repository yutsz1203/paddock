"""The demo: a chat box, its citations, and a plain statement of what it holds.

Run it with the API already up::

    uv run paddock serve                       # :8000
    uv run streamlit run app/streamlit_app.py  # :8501

## This file is widgets only

Everything that can be wrong without looking wrong lives under `paddock.ui` and is
tested: reading the stream, building the banner, choosing a language. What is left
here is layout, and `tests/ui/test_streamlit_app.py` drives it headlessly.

## Two sentences a visitor gets before they ask anything

The **data range** and the **scope**, side by side at the top. A system whose main
claim is that it refuses to answer what it cannot support has to say what it was
given and what it cannot do yet — otherwise a correct refusal reads as a broken
feature (ADR-004, and T22 in `tasks/plan.md`).

## The toggle changes the interface, not the answer

The model answers in the language of the question (`paddock.agent.prompts`). A
caption under the toggle says so, or setting 中文 and asking in English looks like a
bug rather than the intended behaviour.
"""

from __future__ import annotations

import streamlit as st

from paddock.config import get_settings
from paddock.ui.client import ApiClient, ApiError
from paddock.ui.stream import SourceCard
from paddock.ui.text import (
    LANGUAGES,
    Language,
    data_banner,
    route_name,
    source_kind_name,
    strings,
)

st.set_page_config(page_title="paddock", page_icon="🐎", layout="centered")


def api() -> ApiClient:
    """The API client for this session.

    Held in session state rather than rebuilt on every rerun, so the connection pool
    survives a keystroke. It is also the seam the headless tests use to hand the app
    a transport that answers without a server.
    """
    if "api" not in st.session_state:
        st.session_state.api = ApiClient.at(get_settings().ui_api_base_url)
    client: ApiClient = st.session_state.api
    return client


def language() -> Language:
    """Whichever the toggle is set to. English until someone changes it."""
    chosen = st.session_state.get("language_label", LANGUAGES["en"])
    for code, label in LANGUAGES.items():
        if label == chosen:
            return code
    return "en"


def render_header(table_language: Language) -> None:
    table = strings(table_language)

    heading, toggle = st.columns([3, 2], vertical_alignment="bottom")
    with heading:
        st.title(table.title)
        st.caption(table.tagline)
    with toggle:
        st.segmented_control(
            table.title,
            options=list(LANGUAGES.values()),
            default=LANGUAGES[table_language],
            key="language_label",
            label_visibility="collapsed",
        )
        st.caption(table.language_note)


def render_scope(table_language: Language) -> None:
    """The data range and the stated limit, together.

    Fetched on every rerun rather than cached. It is one local request, and a banner
    that keeps claiming a range after `ingest since` has moved it is the failure the
    banner exists to prevent.
    """
    table = strings(table_language)

    try:
        coverage = api().coverage()
    except ApiError as error:
        st.error(str(error))
        return

    st.info(
        data_banner(
            table_language,
            seasons=coverage.seasons,
            last_date=coverage.last_date,
            meetings=coverage.meetings,
        )
    )
    with st.expander(table.scope_title):
        st.markdown(table.scope_body)


def render_sources(table_language: Language, sources: list[SourceCard]) -> None:
    """One expander per citation, holding the text the answer was built from.

    The reference is in the label — `incident_comment:77` — so a reader who wants
    the row can find it without opening the card.
    """
    table = strings(table_language)
    st.markdown(f"**{table.sources_title}**")
    for source in sources:
        kind = source_kind_name(table_language, source.kind)
        with st.expander(f"[{source.marker}] {kind} · {source.reference}"):
            st.markdown(source.text)


def render_history(table_language: Language) -> None:
    for turn in st.session_state.get("turns", []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("sources"):
                render_sources(table_language, turn["sources"])
            if turn.get("footer"):
                st.caption(turn["footer"])


def ask(table_language: Language, question: str) -> None:
    """Send one question and draw the answer as it arrives."""
    table = strings(table_language)
    turns = st.session_state.setdefault("turns", [])
    turns.append({"role": "user", "text": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with api().ask(question) as stream:
                with st.spinner(table.thinking):
                    text = str(st.write_stream(stream))
                sources, route = stream.sources, stream.route
                abstained, failure = stream.abstained, stream.error
        except ApiError as error:
            st.error(str(error))
            turns.append({"role": "assistant", "text": str(error)})
            return

        if failure is not None:
            message = (
                table.error_llm
                if failure == "llm_not_configured"
                else table.error_generic.format(name=failure)
            )
            st.error(message)
            turns.append({"role": "assistant", "text": message})
            return

        footer = f"{table.route_title}: {route_name(table_language, route or '')}"
        if abstained:
            st.markdown(table.abstained_note)
            text = f"{text}\n\n{table.abstained_note}"
        else:
            render_sources(table_language, sources)
        st.caption(footer)

    turns.append(
        {
            "role": "assistant",
            "text": text,
            "sources": [] if abstained else sources,
            "footer": footer,
        }
    )


def render_examples(table_language: Language) -> str | None:
    """Three questions from spec §1, one click from an answer.

    Returns the one that was clicked, if any. A demo whose first screen is an empty
    box asks a visitor to invent a horse name they do not have.
    """
    table = strings(table_language)
    if st.session_state.get("turns"):
        return None

    st.caption(table.examples_title)
    for index, example in enumerate(table.examples):
        if st.button(example, key=f"example_{table_language}_{index}", width="stretch"):
            return example
    return None


chosen_language = language()
render_header(chosen_language)
render_scope(chosen_language)
render_history(chosen_language)

typed = st.chat_input(strings(chosen_language).input_placeholder)
clicked = render_examples(chosen_language)

if typed or clicked:
    ask(chosen_language, typed or clicked or "")
    st.rerun()
