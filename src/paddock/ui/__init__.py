"""Everything the Streamlit demo needs that is not a widget.

The UI is split in two. This package holds the parts that can be wrong silently —
reading the SSE stream, building the banner, choosing a language — and they are
tested. `app/streamlit_app.py` holds the widgets, and is verified by looking at it.

The split also keeps `streamlit` out of the test run. It is an optional extra
(`uv sync --extra ui`), so CI installs neither it nor its dependency tree to check
that a `sources` event is not dropped.
"""
