import base64
import logging
import os
from pathlib import Path
import cv2
import joblib
import numpy as np

logger = logging.getLogger("asl.model")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "app" / "asl_model.joblib"

_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
_model = None


def get_hog_descriptor():
    return _HOG


def extract_hog_feature(img_gray):
    """Resizes grayscale image to 64x64 and computes 1764-dim HOG feature vector."""
    if img_gray is None:
        return None
    resized = cv2.resize(img_gray, (64, 64))
    feat = _HOG.compute(resized)
    return feat.flatten() if feat is not None else None


def load_model():
    """Load the trained ASL classification model if available."""
    global _model
    if _model is not None:
        return _model
    if MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            logger.info(f"Loaded ASL model from {MODEL_PATH}")
            return _model
        except Exception as e:
            logger.error(f"Failed to load ASL model: {e}")
            return None
    return None


def is_model_ready():
    return load_model() is not None


def predict_image(image_data, top_k=5):
    """
    Accepts:
      - base64 data URL string (e.g. 'data:image/jpeg;base64,...' or raw base64)
      - bytes buffer
      - numpy ndarray (grayscale or BGR)
    Returns list of dicts: [{'label': 'A', 'score': 98.4}, ...]
    """
    clf = load_model()
    if clf is None:
        return []

    try:
        img_gray = None
        if isinstance(image_data, str):
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_data, bytes):
            np_arr = np.frombuffer(image_data, np.uint8)
            img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_data, np.ndarray):
            if len(image_data.shape) == 3:
                img_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image_data

        if img_gray is None or img_gray.size == 0:
            return []

        feat = extract_hog_feature(img_gray)
        if feat is None:
            return []

        feat = feat.reshape(1, -1)
        classes = clf.classes_

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(feat)[0]
        else:
            # Fallback for models without predict_proba (decision function softmax)
            scores = clf.decision_function(feat)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

        ranked = sorted(zip(classes, probs), key=lambda x: -x[1])[:top_k]
        return [{"label": str(lbl), "score": round(float(p) * 100, 1)} for lbl, p in ranked]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return []
