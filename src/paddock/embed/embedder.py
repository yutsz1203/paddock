"""The embedding model: BAAI/bge-m3, local, CPU.

## Why this model

The corpus is English — stewards write in English — and half the questions will be
Chinese. bge-m3 puts both languages in one space, so a Chinese question retrieves an
English comment without translating anything at query time. It is also 1024-dim
rather than 384, which is the reason `chunks.embedding` is `vector(1024)`.

Running it locally rather than through an API is a cost decision that happens to be
a correctness one too: no key to leak, no per-call budget to blow during a backfill
of 12,000 chunks, and the demo box keeps working when a free tier expires.

## No query/document asymmetry

E5-family models require "query: " and "passage: " prefixes and score badly without
them. bge-m3 does not — it is trained for symmetric similarity — so there is one
`embed()` here and not a `embed_query`/`embed_documents` pair. That is a fact about
this model, not a simplification: swapping in e5 later means adding the prefixes,
which is why the `Embedder` protocol is what callers depend on.

## Loading is deferred and shared

`SentenceTransformer(...)` reads ~2.2 GB off disk and takes seconds. `get_embedder`
caches one instance per process, and the model is not touched until the first
`embed()` — so importing this module in a test that never embeds anything stays free.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from paddock.config import get_settings
from paddock.db.models import EMBEDDING_DIM

if TYPE_CHECKING:  # pragma: no cover - import cost only paid at runtime
    from sentence_transformers import SentenceTransformer

__all__ = ["EMBEDDING_DIM", "BgeM3Embedder", "Embedder", "get_embedder"]


class Embedder(Protocol):
    """What the rest of paddock needs from an embedding model.

    Narrow on purpose: the store, the retrieval tools and the eval harness all
    depend on this and never on sentence-transformers, so a test can substitute a
    deterministic fake and the demo can fall back to a smaller model on ARM without
    any caller changing.
    """

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one unit-length vector per text, in the order given."""
        ...


class BgeM3Embedder:
    """`Embedder` backed by sentence-transformers, pinned to CPU."""

    def __init__(self, model_name: str | None = None, *, batch_size: int = 16) -> None:
        settings = get_settings()
        # The column is vector(1024) in a migration. Catching a mismatch here turns
        # "swapped EMBEDDING_MODEL in .env for the 384-dim ARM fallback" into one
        # clear error at startup, rather than an opaque pgvector failure thousands
        # of comments into a backfill.
        if settings.embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                f"embedding_dim={settings.embedding_dim} but chunks.embedding is "
                f"vector({EMBEDDING_DIM}) — migrate the column before changing the model"
            )

        self.model_name = model_name or settings.embedding_model
        self.dim = settings.embedding_dim
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            # Imported here, not at module scope: torch costs ~2 s to import and is
            # an optional extra (`uv sync --extra embed`), so a process that only
            # parses HTML should never pay for it.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        # Normalised at encode time so cosine distance is a dot product, and so the
        # HNSW index built with `vector_cosine_ops` compares like with like.
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """The process-wide embedder. Loads the model on first `embed()`, not here."""
    return BgeM3Embedder()
