"""Coach schema: vocabulary metadata on signs, real reference clips/videos, learners, richer attempts.

Legacy synthetic frames are kept but tagged frames_source='synthetic' (dev fallback only).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-27
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("signs") as b:
        b.add_column(sa.Column("gloss", sa.String))
        b.add_column(sa.Column("category", sa.String))
        b.add_column(sa.Column("kind", sa.String))
        b.add_column(sa.Column("hands", sa.String))
        b.add_column(sa.Column("description", sa.Text))
        b.add_column(sa.Column("hint_tips", sa.JSON))
        b.add_column(sa.Column("frames_source", sa.String))
        b.add_column(sa.Column("created_at", sa.DateTime, server_default=sa.func.now()))
        b.alter_column("frames", existing_type=sa.JSON, nullable=True)
        b.create_index("ix_signs_gloss", ["gloss"])
        b.create_index("ix_signs_category", ["category"])

    # Classify legacy rows. Old 'alphabet'/'common' rows were procedurally generated templates.
    op.execute("""
        UPDATE signs SET
          category = CASE source WHEN 'alphabet' THEN 'alphabet' WHEN 'common' THEN 'word'
                                 WHEN 'user' THEN 'user' ELSE 'phrase' END,
          frames_source = CASE source WHEN 'alphabet' THEN 'synthetic' WHEN 'common' THEN 'synthetic'
                                      WHEN 'user' THEN 'legacy_user' ELSE 'legacy_v0' END,
          gloss = UPPER(REPLACE(label, ' ', '-')),
          kind = CASE WHEN source = 'alphabet' AND UPPER(label) NOT IN ('J', 'Z') THEN 'static' ELSE 'dynamic' END
    """)

    op.create_table(
        "reference_clips",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sign_id", sa.Integer, sa.ForeignKey("signs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("signer_id", sa.String, nullable=False),
        sa.Column("split", sa.String),
        sa.Column("landmarks_path", sa.String, nullable=False),
        sa.Column("video_path", sa.String),
        sa.Column("fps", sa.Float),
        sa.Column("num_frames", sa.Integer),
        sa.Column("start_ms", sa.Float),
        sa.Column("end_ms", sa.Float),
        sa.Column("quality", sa.JSON),
        sa.Column("license", sa.String),
        sa.Column("attribution", sa.String),
        sa.Column("source_url", sa.String),
        sa.Column("is_canonical", sa.Boolean, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    for col in ("sign_id", "source", "signer_id", "split"):
        op.create_index(f"ix_reference_clips_{col}", "reference_clips", [col])

    op.create_table(
        "reference_videos",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sign_id", sa.Integer, sa.ForeignKey("signs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("clip_id", sa.Integer, sa.ForeignKey("reference_clips.id", ondelete="SET NULL")),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("duration_ms", sa.Float),
        sa.Column("license", sa.String),
        sa.Column("attribution", sa.String),
        sa.Column("source_url", sa.String),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "learners",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("settings", sa.JSON),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("last_seen", sa.DateTime, server_default=sa.func.now()),
    )

    with op.batch_alter_table("attempts") as b:
        b.add_column(sa.Column("learner_id", sa.String))
        b.add_column(sa.Column("mode", sa.String))
        b.add_column(sa.Column("components", sa.JSON))
        b.add_column(sa.Column("hints", sa.JSON))
        b.add_column(sa.Column("duration_ms", sa.Float))
        b.add_column(sa.Column("model_version", sa.String))
        b.create_index("ix_attempts_learner_id", ["learner_id"])


def downgrade():
    with op.batch_alter_table("attempts") as b:
        b.drop_index("ix_attempts_learner_id")
        for col in ("model_version", "duration_ms", "hints", "components", "mode", "learner_id"):
            b.drop_column(col)
    op.drop_table("learners")
    op.drop_table("reference_videos")
    op.drop_table("reference_clips")
    with op.batch_alter_table("signs") as b:
        b.drop_index("ix_signs_category")
        b.drop_index("ix_signs_gloss")
        for col in ("created_at", "frames_source", "hint_tips", "description", "hands", "kind", "category", "gloss"):
            b.drop_column(col)
