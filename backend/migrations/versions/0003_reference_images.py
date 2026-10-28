"""Reference photos: one real photo of the handshape per sign (letters without a video), with license/attribution.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-28
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "reference_images",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sign_id", sa.Integer, sa.ForeignKey("signs.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("clip_id", sa.Integer, sa.ForeignKey("reference_clips.id", ondelete="SET NULL")),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("license", sa.String),
        sa.Column("attribution", sa.String),
        sa.Column("source_url", sa.String),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("reference_images")
