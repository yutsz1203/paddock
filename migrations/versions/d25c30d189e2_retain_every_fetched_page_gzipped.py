"""retain every fetched page, gzipped

Revision ID: d25c30d189e2
Revises: c995bae9e06f
Create Date: 2026-08-17 13:09:25.813139

Checkpoint A step 6 priced the alternative: keeping only `source_url` makes any
parser fix after the backfill a re-scrape of both seasons — ~3,700 requests, an hour
at 1 req/s, and the risk that HKJC's markup moved or the page is gone. Unlike every
other reversible decision on that list, this one cannot be retrofitted; a page not
kept is not recoverable. ~10 MB gzipped for two seasons buys re-parses for free.

The downgrade drops the archive, and that loss is real rather than mechanical — the
pages cannot be rebuilt from anything else in the database.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d25c30d189e2"
down_revision: str | Sequence[str] | None = "c995bae9e06f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fetched_pages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("body_gz", sa.LargeBinary(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # (url, fetched_at) rather than url alone: every read is "the newest version of
    # this URL", which this index answers without sorting the versions of a URL.
    op.create_index("ix_fetched_pages_url_time", "fetched_pages", ["url", "fetched_at"])


def downgrade() -> None:
    """Downgrade schema. Irreversible in substance: the archived pages are gone."""
    op.drop_index("ix_fetched_pages_url_time", table_name="fetched_pages")
    op.drop_table("fetched_pages")
