"""Trained recognisers, loaded once. Model dirs: $HANDSPEAK_MODELS_DIR/<task>/<version>/ with a CURRENT pointer.

Each version dir holds model.joblib, metrics.json and (for in-browser use) browser.json.
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from . import features as fx
from .config import get_settings

logger = logging.getLogger("asl.recognition")


class NoHandError(ValueError):
    pass


def current_dir(task: str) -> Path | None:
    root = get_settings().models_dir / task
    ptr = root / "CURRENT"
    if not ptr.exists():
        return None
    d = root / ptr.read_text(encoding="utf-8").strip()
    return d if d.is_dir() else None


def calibrate(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling applied to probabilities (== softmax(logits / T))."""
    logp = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


class StaticModel:
    """Alphabet hand-shape classifier on FEATURE_SPEC §9 static vectors."""

    def __init__(self, path: Path):
        art = joblib.load(path / "model.joblib")
        if art.get("spec_version") != fx.SPEC_VERSION:
            raise ValueError(f"model spec {art.get('spec_version')} != feature spec {fx.SPEC_VERSION}")
        self.estimator = art["estimator"]
        self.classes = list(art["classes"])
        self.temperature = float(art.get("temperature", 1.0))
        self.reject_threshold = float(art.get("reject_threshold", 0.0))
        self.version = art["version"]
        self.kind = art["kind"]
        self.dir = path
        mp = path / "metrics.json"
        self.metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}

    def proba(self, X: np.ndarray) -> np.ndarray:
        p = self.estimator.predict_proba(X)
        order = [list(self.estimator.classes_).index(c) for c in self.classes]
        return calibrate(p[:, order], self.temperature)

    def predict_frames(self, frames, left_handed=False):
        """Average calibrated probabilities over every frame that shows a hand."""
        vecs, idx = fx.static_vectors(frames, left_handed)
        if len(vecs) == 0:
            raise NoHandError("No hand detected. Keep your hand inside the frame and try again.")
        p = self.proba(vecs).mean(axis=0)
        return p, len(vecs)

    def summary(self):
        m = self.metrics
        return {"ready": True, "version": self.version, "type": self.kind, "classes": len(self.classes),
                "eval_protocol": m.get("eval_protocol"), "accuracy": m.get("test", {}).get("accuracy"),
                "macro_f1": m.get("test", {}).get("macro_f1"), "test_signers": m.get("signers", {}).get("test"),
                "trained_at": m.get("created_at"), "warning": m.get("warning"),
                "verified": bool(m.get("verified")), "video_signers": len(m.get("video_signers") or []),
                "loso_mean": (m.get("leave_one_signer_out") or {}).get("mean")}


class Models:
    def __init__(self):
        self.alphabet: StaticModel | None = None
        self.errors: dict[str, str] = {}

    def load(self):
        self.alphabet, self.errors = None, {}
        d = current_dir("alphabet")
        if d is None:
            self.errors["alphabet"] = "not trained (run scripts.train_alphabet)"
            return
        try:
            self.alphabet = StaticModel(d)
            logger.info("Loaded alphabet model %s", self.alphabet.version)
        except Exception as e:  # noqa: BLE001 - keep the API up without the model
            self.errors["alphabet"] = str(e)
            logger.error("Alphabet model failed to load: %s", e)

    def status(self):
        a = self.alphabet.summary() if self.alphabet else {"ready": False, "error": self.errors.get("alphabet")}
        return {"alphabet": a, "words": {"ready": False, "error": "dynamic word model arrives in phase P3"}}


models = Models()
