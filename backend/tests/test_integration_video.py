"""Integration: real held-out-signer videos -> MediaPipe -> features -> trained word model -> API.

Uses local data that is not in the repo (ASL Citizen clips, the real asl.db, a trained word model) and is skipped
when any of it is missing, so the suite stays portable.
"""
import sqlite3
from pathlib import Path

import pytest
from dotenv import dotenv_values

BACKEND = Path(__file__).resolve().parent.parent
N_CLIPS = 6


def _real_paths():
    env = dotenv_values(BACKEND / ".env")
    store = Path(env.get("HANDSPEAK_STORE_DIR") or BACKEND / "data")
    models = Path(env.get("HANDSPEAK_MODELS_DIR") or BACKEND / "models") / "words"
    db = BACKEND / "asl.db"
    ptr = models / "CURRENT"
    if not (db.exists() and ptr.exists() and store.is_dir()):
        return None
    return store, models / ptr.read_text().strip(), db


@pytest.fixture(scope="module")
def real():
    paths = _real_paths()
    if paths is None:
        pytest.skip("no local ASL Citizen data / trained word model")
    from app import mediapipe_models
    if not mediapipe_models.available():
        pytest.skip("MediaPipe models not downloaded")
    store, model_dir, db = paths
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT s.label, r.video_path FROM reference_clips r JOIN signs s ON s.id = r.sign_id "
                       "WHERE r.source = 'asl_citizen' AND r.split = 'test' AND r.video_path IS NOT NULL "
                       "GROUP BY s.label ORDER BY s.label LIMIT ?", (N_CLIPS,)).fetchall()
    con.close()
    clips = [(label, store / vp) for label, vp in rows if (store / vp).exists()]
    if len(clips) < N_CLIPS:
        pytest.skip("not enough local test-split videos")
    from app.word_model import WordModel
    return WordModel(model_dir), clips


def test_real_videos_end_to_end(client, real, monkeypatch):
    from app.extract import video_frames
    from app.recognition import models
    wm, clips = real
    monkeypatch.setattr(models, "words", wm)
    hits = 0
    for label, video in clips:
        frames, fps = video_frames(video, max_side=720)
        assert fps > 0 and any(f["hands"] for f in frames)
        r = client.post("/api/recognize", json={"frames": frames, "mode": "word", "top_k": 5})
        assert r.status_code == 200, r.text
        top5 = [m["label"] for m in r.json()["matches"]]
        hits += label in top5
    # Held-out signers, one clip per sign: the reported top-5 accuracy is ~95%, so allow some misses.
    assert hits >= N_CLIPS - 2, f"true sign in top-5 for only {hits}/{N_CLIPS} real clips"
