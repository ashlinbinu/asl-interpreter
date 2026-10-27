"""Pinned MediaPipe Tasks model files, shared by the backend and the browser (same URLs, same bytes)."""
from .config import get_settings

TASKS_VISION_VERSION = "0.10.14"   # must match frontend package.json and requirements.txt (mediapipe)

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
}


def model_path(name: str):
    return get_settings().models_dir / "mediapipe" / name


def available() -> bool:
    return all(model_path(n).exists() for n in MODELS)
