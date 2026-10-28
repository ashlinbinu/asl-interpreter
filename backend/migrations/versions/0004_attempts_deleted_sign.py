"""Attempts outlive deleted custom signs: keep the sign's label so progress can still show it.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-28
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("attempts") as b:
        b.add_column(sa.Column("deleted_sign_label", sa.String))


def downgrade():
    with op.batch_alter_table("attempts") as b:
        b.drop_column("deleted_sign_label")
