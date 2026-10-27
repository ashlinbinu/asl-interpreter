"""Legacy baseline: the schema the pre-coach app created with create_all().

Existing databases without alembic_version are stamped at this revision by app.db.init_db().

Revision ID: 0001
Revises:
Create Date: 2026-10-27
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "signs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("label", sa.String, nullable=False),
        sa.Column("frames", sa.JSON, nullable=False),
        sa.Column("video_url", sa.String),
        sa.Column("source", sa.String),
    )
    op.create_index("ix_signs_label", "signs", ["label"])
    op.create_table(
        "attempts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sign_id", sa.Integer),
        sa.Column("score", sa.Float),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_attempts_sign_id", "attempts", ["sign_id"])


def downgrade():
    op.drop_table("attempts")
    op.drop_table("signs")
