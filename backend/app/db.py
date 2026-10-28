"""SQLAlchemy models and session. The schema is owned by Alembic (backend/migrations); call init_db() once."""
import logging

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, func, inspect, text)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import BACKEND_DIR, get_settings

logger = logging.getLogger("asl.db")

DATABASE_URL = get_settings().database_url
_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_kwargs)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class Sign(Base):
    """A vocabulary entry (letter, word, phrase). Real reference data lives in ReferenceClip."""
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True)
    label = Column(String, index=True, nullable=False)
    gloss = Column(String, index=True)
    category = Column(String, index=True)          # alphabet | word | phrase | user
    kind = Column(String)                          # static | dynamic
    hands = Column(String)                         # one | two
    description = Column(Text)
    hint_tips = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    # Legacy (pre-coach) columns, used only by the dev synthetic fallback.
    frames = Column(JSON, nullable=True)
    frames_source = Column(String)                 # synthetic | legacy_user | legacy_v0
    video_url = Column(String)
    source = Column(String)


class ReferenceClip(Base):
    __tablename__ = "reference_clips"
    id = Column(Integer, primary_key=True)
    sign_id = Column(Integer, ForeignKey("signs.id", ondelete="CASCADE"), index=True, nullable=False)
    source = Column(String, index=True, nullable=False)   # own | asl_citizen | wlasl | kaggle_alphabet | ...
    signer_id = Column(String, index=True, nullable=False)
    split = Column(String, index=True)                    # train | val | test (assigned by signer)
    landmarks_path = Column(String, nullable=False)       # relative to store_dir
    video_path = Column(String)                           # relative to store_dir
    fps = Column(Float)
    num_frames = Column(Integer)
    start_ms = Column(Float)
    end_ms = Column(Float)
    quality = Column(JSON)
    license = Column(String)
    attribution = Column(String)
    source_url = Column(String)
    is_canonical = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())


class ReferenceVideo(Base):
    __tablename__ = "reference_videos"
    id = Column(Integer, primary_key=True)
    sign_id = Column(Integer, ForeignKey("signs.id", ondelete="CASCADE"), unique=True, nullable=False)
    clip_id = Column(Integer, ForeignKey("reference_clips.id", ondelete="SET NULL"))
    path = Column(String, nullable=False)                 # relative to media_dir
    width = Column(Integer)
    height = Column(Integer)
    duration_ms = Column(Float)
    license = Column(String)
    attribution = Column(String)
    source_url = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class ReferenceImage(Base):
    """A real photo of the handshape (letters that have no video), copied from a dataset whose license allows it."""
    __tablename__ = "reference_images"
    id = Column(Integer, primary_key=True)
    sign_id = Column(Integer, ForeignKey("signs.id", ondelete="CASCADE"), unique=True, nullable=False)
    clip_id = Column(Integer, ForeignKey("reference_clips.id", ondelete="SET NULL"))
    path = Column(String, nullable=False)                 # relative to media_dir
    width = Column(Integer)
    height = Column(Integer)
    license = Column(String)
    attribution = Column(String)
    source_url = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class Learner(Base):
    __tablename__ = "learners"
    id = Column(String, primary_key=True)                 # anonymous id from the browser
    settings = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    last_seen = Column(DateTime, server_default=func.now())


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True)
    sign_id = Column(Integer, index=True)
    score = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    learner_id = Column(String, index=True)
    mode = Column(String)                                 # practice | sentence | recognize
    components = Column(JSON)
    hints = Column(JSON)
    duration_ms = Column(Float)
    model_version = Column(String)
    deleted_sign_label = Column(String)                   # set when its (custom) sign is deleted


def migrate(url: str, revision: str = "head"):
    """Bring a database to `revision` (stamps a pre-Alembic asl.db as the legacy baseline first)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["configure_logger"] = False
    eng = create_engine(url)
    try:
        tables = set(inspect(eng).get_table_names())
    finally:
        eng.dispose()
    if "signs" in tables and "alembic_version" not in tables:
        logger.info("Existing pre-migration database found; stamping legacy baseline.")
        command.stamp(cfg, "0001")
    command.upgrade(cfg, revision)


def init_db():
    migrate(DATABASE_URL)


def check_connection():
    """Database reachability and counts for /api/status."""
    kind = "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql"
    try:
        with Session() as s:
            s.execute(text("SELECT 1"))
            cats = {c or "unknown": n for c, n in s.query(Sign.category, func.count(Sign.id)).group_by(Sign.category)}
            clips = {src: n for src, n in s.query(ReferenceClip.source, func.count(ReferenceClip.id))
                     .group_by(ReferenceClip.source)}
            return {
                "connected": True,
                "database_type": kind,
                "signs_count": sum(cats.values()),
                "categories": cats,
                "reference_clips": clips,
                "reference_videos": s.query(ReferenceVideo).count(),
                "reference_images": s.query(ReferenceImage).count(),
                "attempts_count": s.query(Attempt).count(),
            }
    except Exception as e:  # noqa: BLE001 - status must never raise
        logger.error("Database connection error: %s", e)
        return {"connected": False, "error": str(e), "database_type": kind, "signs_count": 0, "attempts_count": 0}
