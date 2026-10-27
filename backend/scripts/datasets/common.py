"""Shared ingest helpers: every source ends up as one NPZ per clip + one reference_clips row."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.clipstore import quality, save_clip  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import ReferenceClip, Session, Sign, init_db  # noqa: E402
from app.registry import ensure_alphabet, norm  # noqa: E402


def setup():
    init_db()
    ensure_alphabet()


def find_sign(s, label: str, category: str | None = None):
    q = s.query(Sign).filter(Sign.category == category) if category else s.query(Sign)
    for row in q:
        if norm(row.label) == norm(label):
            return row
    return None


def existing_keys(source: str) -> set:
    """landmarks_path values already ingested for a source (makes re-runs incremental)."""
    with Session() as s:
        return {p for (p,) in s.query(ReferenceClip.landmarks_path).filter(ReferenceClip.source == source)}


def add_clip(s, sign, frames, *, source, signer_id, rel_path, license, attribution=None, source_url=None,
             video_path=None, fps=None, split=None, extra_quality=None):
    """Write the NPZ (path relative to the store dir) and stage a ReferenceClip row on session `s`."""
    store = get_settings().store_dir
    meta = {"label": sign.label, "source": source, "signer_id": signer_id, "license": license}
    save_clip(store / rel_path, frames, meta)
    ts = [f["t"] for f in frames] or [0.0]
    q = quality(frames)
    if extra_quality:
        q.update(extra_quality)
    clip = ReferenceClip(sign_id=sign.id, source=source, signer_id=signer_id, split=split, landmarks_path=rel_path,
                         video_path=video_path, fps=fps, num_frames=len(frames), start_ms=min(ts), end_ms=max(ts),
                         quality=q, license=license, attribution=attribution, source_url=source_url)
    s.add(clip)
    return clip
