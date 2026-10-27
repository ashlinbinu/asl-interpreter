"""Offline landmark extraction with MediaPipe Tasks (same models as the browser) -> RawFrames."""
import logging
from pathlib import Path

import cv2
import numpy as np

from .features import POSE_N
from .mediapipe_models import model_path

logger = logging.getLogger("asl.extract")


def _r(v):
    return round(float(v), 5)


class Extractor:
    """Hand (2) + Pose landmarkers. mode='image' for photos, 'video' for clips (one instance per clip)."""

    def __init__(self, mode="video", min_hand_conf=0.5, pose=True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        rm = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        self.mode = mode
        hand_path, pose_path = model_path("hand_landmarker.task"), model_path("pose_landmarker_lite.task")
        if not hand_path.exists():
            raise FileNotFoundError(f"{hand_path} missing. Run: python -m scripts.download_models")
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path)), running_mode=rm, num_hands=2,
            min_hand_detection_confidence=min_hand_conf, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        self.pose = None
        if pose and pose_path.exists():
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_path)), running_mode=rm, num_poses=1))

    def close(self):
        self.hands.close()
        if self.pose:
            self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def frame(self, bgr: np.ndarray, t_ms: float = 0.0) -> dict:
        h, w = bgr.shape[:2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = int(round(t_ms))
        hr = self.hands.detect_for_video(image, ts) if self.mode == "video" else self.hands.detect(image)
        hands = []
        for lm, hd in zip(hr.hand_landmarks, hr.handedness):
            hands.append({"handedness": hd[0].category_name, "score": _r(hd[0].score),
                          "lm": [[_r(p.x), _r(p.y), _r(p.z)] for p in lm]})
        pose = None
        if self.pose:
            pr = self.pose.detect_for_video(image, ts) if self.mode == "video" else self.pose.detect(image)
            if pr.pose_landmarks:
                pose = [[_r(p.x), _r(p.y), _r(p.z), _r(p.visibility)] for p in pr.pose_landmarks[0][:POSE_N]]
        return {"t": round(t_ms, 3), "w": w, "h": h, "hands": hands, "pose": pose}


def image_frame(bgr: np.ndarray, pad: int = 0) -> dict:
    """Landmarks for one photo (optionally padded with a border so tight crops still detect)."""
    if pad:
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    with Extractor(mode="image") as ex:
        return ex.frame(bgr)


def video_frames(path: Path, start_ms: float = 0.0, end_ms: float | None = None, max_side: int = 960):
    """All frames of a clip (optionally a time range) -> (RawFrames, fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    frames, i = [], 0
    with Extractor(mode="video") as ex:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = i * 1000.0 / fps
            i += 1
            if t < start_ms:
                continue
            if end_ms is not None and t > end_ms:
                break
            hh, ww = img.shape[:2]
            if max(hh, ww) > max_side:
                k = max_side / max(hh, ww)
                img = cv2.resize(img, (int(ww * k), int(hh * k)), interpolation=cv2.INTER_AREA)
            frames.append(ex.frame(img, t))
    cap.release()
    return frames, fps
