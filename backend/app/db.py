import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, JSON, create_engine, func, text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger("asl.db")

# Load environment variables from .env in backend directory
BASE_DIR = Path(__file__).resolve().parent.parent
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

DEFAULT_DB_FILE = BASE_DIR / "asl.db"
raw_url = os.getenv("DATABASE_URL", "").strip()

if not raw_url or raw_url.startswith("sqlite:///.") or raw_url == "sqlite:///asl.db":
    DATABASE_URL = f"sqlite:///{DEFAULT_DB_FILE.as_posix()}"
else:
    if raw_url.startswith("postgres://"):
        raw_url = "postgresql://" + raw_url[len("postgres://"):]
    DATABASE_URL = raw_url

engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Sign(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True)
    label = Column(String, index=True, nullable=False)
    frames = Column(JSON, nullable=False)  # 32 x 126 normalized hand features
    video_url = Column(String)
    source = Column(String, default="how2sign")


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True)
    sign_id = Column(Integer, index=True)
    score = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


# Ensure tables are created
Base.metadata.create_all(engine)


def check_connection():
    """Verify database connection and return current statistics."""
    try:
        with Session() as s:
            s.execute(text("SELECT 1"))
            signs_count = s.query(Sign).count()
            attempts_count = s.query(Attempt).count()
            sources = {}
            for r in s.query(Sign.source, func.count(Sign.id)).group_by(Sign.source).all():
                sources[r[0] or "unknown"] = r[1]

            return {
                "connected": True,
                "database_type": "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql",
                "database_url_masked": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else "sqlite://local",
                "signs_count": signs_count,
                "attempts_count": attempts_count,
                "sources": sources,
            }
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return {
            "connected": False,
            "error": str(e),
            "database_type": "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql",
            "signs_count": 0,
            "attempts_count": 0,
            "sources": {},
        }
