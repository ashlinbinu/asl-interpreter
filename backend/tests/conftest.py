"""Test environment: an isolated temp DB, store and models dir (set before any app import)."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
_TMP = Path(tempfile.mkdtemp(prefix="handspeak-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["HANDSPEAK_STORE_DIR"] = str(_TMP / "store")
os.environ["HANDSPEAK_MODELS_DIR"] = str(_TMP / "models")
os.environ["HANDSPEAK_DEV_SYNTHETIC"] = "0"
os.environ["CORS_ORIGINS"] = "http://localhost:5173"

# Reuse real MediaPipe models if they were downloaded (enables the extraction tests).
_real_mp = BACKEND / "models" / "mediapipe"
if _real_mp.is_dir():
    shutil.copytree(_real_mp, _TMP / "models" / "mediapipe")


@pytest.fixture(scope="session")
def tmp_root():
    return _TMP


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


def synthetic_letter_clips(signers=("s1", "s2", "s3"), reps=4, seed=0):
    """TEST ONLY: four fake 'letters' that differ by curl/rotation, from several fake signers."""
    import numpy as np

    from tests import synth
    rng = np.random.default_rng(seed)
    shapes = {"A": (1.0, 0.0), "B": (0.0, 0.0), "C": (0.5, 0.0), "H": (0.0, 1.4)}
    clips, k = [], 0
    for si, signer in enumerate(signers):
        for label, (curl, rot) in shapes.items():
            for r in range(reps):
                frames = []
                for i in range(6):
                    lm = synth.hand((0.5 + 0.02 * si, 0.6), size=0.12 + 0.01 * si,
                                    curl=float(np.clip(curl + rng.normal(0, 0.05), 0, 1)),
                                    rot=rot + rng.normal(0, 0.05), rng=rng, noise=0.002)
                    frames.append({"t": i * 33.3, "w": 640, "h": 480, "pose": None,
                                   "hands": [{"handedness": "Left", "score": 0.9, "lm": lm.tolist()}]})
                clips.append({"frames": frames, "label": label, "signer_id": signer, "clip_id": f"{signer}-{k:04d}",
                              "source": "test"})
                k += 1
    return clips


@pytest.fixture(scope="session")
def trained_alphabet(client, tmp_root):
    from app.recognition import models
    from app.train_static import train
    metrics, out = train(synthetic_letter_clips(), tmp_root / "models" / "alphabet", n_aug=1, log=lambda *_: None)
    models.load()
    return metrics, out
