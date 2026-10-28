"""All runtime configuration, from environment variables or backend/.env. No user-specific paths."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env", override=False)


def _path(name, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def _database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw or raw.startswith("sqlite:///.") or raw == "sqlite:///asl.db":
        return f"sqlite:///{(BACKEND_DIR / 'asl.db').as_posix()}"
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://"):]
    return raw


@dataclass(frozen=True)
class Settings:
    database_url: str
    store_dir: Path          # processed data: landmarks NPZ, media, recordings (keep outside OneDrive)
    models_dir: Path         # trained models + MediaPipe .task files
    data_dir: Path | None    # optional raw image dataset for alphabet ingest
    cors_origins: tuple
    max_body_bytes: int
    max_upload_bytes: int
    dev_synthetic: bool      # allow legacy synthetic templates as a dev-only fallback

    @property
    def media_dir(self) -> Path:
        return self.store_dir / "media"

    @property
    def landmarks_dir(self) -> Path:
        return self.store_dir / "landmarks"

    @property
    def recordings_dir(self) -> Path:
        return self.store_dir / "recordings"


@lru_cache
def get_settings() -> Settings:
    data_dir = os.getenv("DATA_DIR", "").strip()
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return Settings(
        database_url=_database_url(),
        store_dir=_path("HANDSPEAK_STORE_DIR", BACKEND_DIR / "data"),
        models_dir=_path("HANDSPEAK_MODELS_DIR", BACKEND_DIR / "models"),
        data_dir=Path(data_dir).expanduser() if data_dir else None,
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        max_body_bytes=int(os.getenv("HANDSPEAK_MAX_BODY_BYTES", str(8 * 1024 * 1024))),
        max_upload_bytes=int(os.getenv("HANDSPEAK_MAX_UPLOAD_BYTES", str(60 * 1024 * 1024))),
        dev_synthetic=os.getenv("HANDSPEAK_DEV_SYNTHETIC", "0").strip().lower() in ("1", "true", "yes"),
    )
