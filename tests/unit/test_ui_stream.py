"""Reading `/ask` back: SSE lines in, an answer out.

Streamlit is not imported here, and that is the reason this code lives under
`src/paddock/ui` rather than inside `app/streamlit_app.py`. The parsing is the part
that can be wrong in a way nobody sees — a dropped `sources` event renders as an
answer with no citations, which is exactly the failure the whole project exists to
prevent — so it is tested, and the widget layer above it is not.
"""

from __future__ import annotations

import pytest

from paddock.ui.stream import AnswerStream, parse_sse

CITED = "It was hampered near the 800 [S1]."


def _sse(*blocks: str) -> list[str]:
    """Split a wire body into lines the way `httpx.iter_lines` does."""
    return "".join(blocks).split("\n")


def _event(name: str, data: str) -> str:
    return f"event: {name}\ndata: {data}\n\n"


# ── Parsing ─────────────────────────────────────────────────────────────────────


def test_an_event_block_becomes_a_name_and_a_payload() -> None:
    events = list(parse_sse(_sse(_event("token", '{"text": "Hello "}'))))

    assert events == [("token", {"text": "Hello "})]


def test_blocks_are_kept_in_order() -> None:
    events = list(
        parse_sse(
            _sse(
                _event("token", '{"text": "a"}'),
                _event("token", '{"text": "b"}'),
                _event("done", '{"route": "vector"}'),
            )
        )
    )

    assert [name for name, _ in events] == ["token", "token", "done"]


def test_a_payload_with_chinese_survives_the_trip() -> None:
    """The API sends `ensure_ascii=False`, so the bytes are UTF-8 and not escapes."""
    events = list(parse_sse(_sse(_event("token", '{"text": "受阻"}'))))

    assert events[0][1]["text"] == "受阻"


def test_a_trailing_block_with_no_blank_line_is_not_lost() -> None:
    """A stream that ends without its final newline still carries a `done`, and
    dropping it would silently strip the route from the last answer of a session."""
    events = list(parse_sse(_sse('event: done\ndata: {"route": "sql"}')))

    assert events == [("done", {"route": "sql"})]


def test_a_keepalive_comment_is_ignored() -> None:
    """Caddy and uvicorn both emit `:` lines. One must not become an event."""
    events = list(parse_sse(_sse(": keepalive\n\n", _event("token", '{"text": "a"}'))))

    assert events == [("token", {"text": "a"})]


# ── Collecting ──────────────────────────────────────────────────────────────────


def _full_stream() -> list[str]:
    return _sse(
        _event("token", '{"text": "It was hampered "}'),
        _event("token", '{"text": "near the 800 [S1]."}'),
        _event(
            "sources",
            '{"sources": [{"marker": "S1", "kind": "comment", "text": "Was hampered.",'
            ' "reference": "incident_comment:77"}]}',
        ),
        _event(
            "done",
            '{"route": "vector", "horse_id": "HK_2024_K570", "horse_name": "SETANTA",'
            ' "abstained": false, "attempts": 1}',
        ),
    )


def test_iterating_yields_the_answer_piece_by_piece() -> None:
    stream = AnswerStream(parse_sse(_full_stream()))

    assert "".join(stream) == CITED


def test_the_sources_are_available_once_the_stream_is_drained() -> None:
    """`st.write_stream` consumes the generator and returns; the cards are rendered
    after it, so they have to survive the iteration."""
    stream = AnswerStream(parse_sse(_full_stream()))
    list(stream)

    assert [source.marker for source in stream.sources] == ["S1"]
    assert stream.sources[0].kind == "comment"
    assert stream.sources[0].reference == "incident_comment:77"


def test_the_route_and_the_abstention_flag_survive_too() -> None:
    stream = AnswerStream(parse_sse(_full_stream()))
    list(stream)

    assert stream.route == "vector"
    assert stream.horse_name == "SETANTA"
    assert stream.abstained is False


def test_reading_sources_before_the_stream_is_drained_is_refused() -> None:
    """Half a stream is not an answer. Returning `[]` here would render a cited
    answer as an uncited one, which is the one thing the UI must never do."""
    stream = AnswerStream(parse_sse(_full_stream()))

    with pytest.raises(RuntimeError):
        _ = stream.sources


def test_an_error_event_ends_the_stream_and_is_readable_after_it() -> None:
    stream = AnswerStream(parse_sse(_sse(_event("error", '{"message": "llm_not_configured"}'))))

    assert list(stream) == []
    assert stream.error == "llm_not_configured"
    assert stream.sources == []


def test_an_error_after_some_tokens_keeps_what_arrived() -> None:
    stream = AnswerStream(
        parse_sse(
            _sse(
                _event("token", '{"text": "It was "}'),
                _event("error", '{"message": "TimeoutError"}'),
            )
        )
    )

    assert "".join(stream) == "It was "
    assert stream.error == "TimeoutError"


def test_a_refusal_is_an_ordinary_answer_with_no_sources() -> None:
    """The API streams "I don't know" through the same three events, so the UI needs
    no second code path — only a flag to label it with."""
    stream = AnswerStream(
        parse_sse(
            _sse(
                _event("token", '{"text": "I have no evidence."}'),
                _event("sources", '{"sources": []}'),
                _event("done", '{"route": "vector", "abstained": true}'),
            )
        )
    )

    assert "".join(stream) == "I have no evidence."
    assert stream.abstained is True
    assert stream.sources == []
