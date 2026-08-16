"""Constructing the embedder — no model is loaded here.

`BgeM3Embedder.__init__` deliberately does no work beyond reading settings, so these
run in milliseconds and without the 2.2 GB download. The one behaviour worth pinning
is the dimension check: `chunks.embedding` is `vector(1024)` in a migration, and a
model that disagrees would otherwise surface as an opaque pgvector error thousands of
comments into a backfill.
"""

from __future__ import annotations

import pytest

from paddock.config import Settings
from paddock.embed.embedder import EMBEDDING_DIM, BgeM3Embedder


def test_the_default_model_matches_the_column_width() -> None:
    assert BgeM3Embedder().dim == EMBEDDING_DIM


def test_a_model_of_the_wrong_width_is_refused_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swapping `EMBEDDING_MODEL` in .env without migrating the column is a
    plausible mistake — e5-small is 384-dim and is the documented ARM fallback."""
    monkeypatch.setattr(
        "paddock.embed.embedder.get_settings",
        lambda: Settings(embedding_model="intfloat/multilingual-e5-small", embedding_dim=384),
    )

    with pytest.raises(ValueError, match="1024"):
        BgeM3Embedder()


def test_the_model_is_not_loaded_until_something_is_embedded() -> None:
    """Importing or constructing must stay free: the API process and the parsers
    both import this module and neither embeds anything."""
    embedder = BgeM3Embedder()

    assert embedder._model is None


def test_embedding_nothing_needs_no_model() -> None:
    """A meeting where every runner ran clean has no comments to embed, and must not
    pay for a model load to discover that."""
    embedder = BgeM3Embedder()

    assert embedder.embed([]) == []
    assert embedder._model is None
