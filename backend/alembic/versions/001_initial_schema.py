"""Initial fraud platform schema.

Cases live in Postgres with transactions and scores. SQLite is not a second
system of record.

Revision ID: 001
"""
from pathlib import Path

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    raw = (Path(__file__).with_name("001_initial.sql")).read_text()
    lines = [line for line in raw.splitlines() if not line.strip().startswith("--")]
    statements = [part.strip() for part in "\n".join(lines).split(";") if part.strip()]
    for statement in statements:
        op.execute(statement)


def downgrade():
    op.execute("DROP TABLE IF EXISTS investigation_cases")
    op.execute("DROP TABLE IF EXISTS fraud_scores")
    op.execute("DROP TABLE IF EXISTS transactions")
    op.execute("DROP TABLE IF EXISTS users")
