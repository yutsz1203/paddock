"""chunks: address by (source, index) so a comment can hold several

Revision ID: c995bae9e06f
Revises: 2e7b2f773e55
Create Date: 2026-08-17 11:35:05.195132

Checkpoint A (ADR-003) measured one-vector-per-comment and found 72% of the corpus
unreachable, so a comment is now embedded one sentence-sized piece at a time. The
address of a chunk gains its position within the comment; `source_id` alone stays
what a citation resolves.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c995bae9e06f"
down_revision: str | Sequence[str] | None = "2e7b2f773e55"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # server_default so this applies to a populated table: every existing chunk is
    # the whole comment, which is piece 0 of it. Dropped afterwards, because the
    # application always states the index — a default would hide a caller that forgot.
    op.add_column(
        "chunks",
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("chunks", "chunk_index", server_default=None)

    op.drop_constraint("uq_chunk_source", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunk_source", "chunks", ["source_type", "source_id", "chunk_index"]
    )


def downgrade() -> None:
    """Downgrade schema.

    Lossy by necessity: the old constraint permits one chunk per comment, so the
    pieces after the first cannot survive it. They are deleted rather than merged —
    re-running `embed_meeting` under the old code would rebuild whole-comment chunks
    anyway, and a silently merged text would be neither the old chunk nor the new one.
    """
    op.execute(sa.text("DELETE FROM chunks WHERE chunk_index > 0"))

    op.drop_constraint("uq_chunk_source", "chunks", type_="unique")
    op.create_unique_constraint("uq_chunk_source", "chunks", ["source_type", "source_id"])
    op.drop_column("chunks", "chunk_index")
