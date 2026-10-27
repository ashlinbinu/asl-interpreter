"""Per-clip landmark storage: RawFrames (FEATURE_SPEC §1) <-> compressed NPZ.

Hands are stored in detection order (not canonical slots) so features can be recomputed under any spec version.
"""
import json
from pathlib import Path

import numpy as np

from .features import POSE_N

_LABELS = {None: 0, "Left": 1, "Right": 2}
_LABELS_INV = {0: None, 1: "Left", 2: "Right"}


def save_clip(path: Path, frames: list, meta: dict) -> Path:
    n = len(frames)
    hands = np.zeros((n, 2, 21, 3), np.float32)
    present = np.zeros((n, 2), bool)
    label = np.zeros((n, 2), np.int8)
    score = np.zeros((n, 2), np.float32)
    pose = np.zeros((n, POSE_N, 4), np.float32)
    pose_present = np.zeros(n, bool)
    t_ms = np.zeros(n, np.float64)
    w = h = 0
    for i, f in enumerate(frames):
        t_ms[i] = float(f.get("t", i * 1000 / 30))
        w, h = int(f.get("w") or w or 640), int(f.get("h") or h or 480)
        for k, hd in enumerate((f.get("hands") or [])[:2]):
            hands[i, k] = np.asarray(hd["lm"], np.float32).reshape(21, 3)
            present[i, k] = True
            label[i, k] = _LABELS.get(hd.get("handedness"), 0)
            score[i, k] = float(hd.get("score") or 0)
        if f.get("pose") is not None:
            p = np.asarray(f["pose"], np.float32)[:POSE_N]
            pose[i, :, :p.shape[1]] = p
            if p.shape[1] < 4:
                pose[i, :, 3] = 1.0
            pose_present[i] = True
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hands=hands, hand_present=present, handedness=label, hand_score=score,
                        pose=pose, pose_present=pose_present, t_ms=t_ms, size=np.array([w, h], np.int32),
                        meta=np.array(json.dumps(meta)))
    return path


def load_clip(path: Path):
    """-> (frames, meta)"""
    d = np.load(path, allow_pickle=False)
    w, h = (int(v) for v in d["size"])
    frames = []
    for i in range(len(d["t_ms"])):
        hands = [{"handedness": _LABELS_INV[int(d["handedness"][i, k])], "score": float(d["hand_score"][i, k]),
                  "lm": d["hands"][i, k].astype(float).tolist()} for k in range(2) if d["hand_present"][i, k]]
        pose = d["pose"][i].astype(float).tolist() if d["pose_present"][i] else None
        frames.append({"t": float(d["t_ms"][i]), "w": w, "h": h, "hands": hands, "pose": pose})
    return frames, json.loads(str(d["meta"]))


def quality(frames: list) -> dict:
    """Simple capture-quality stats used by the recorder and ingest scripts."""
    n = max(len(frames), 1)
    hand = sum(1 for f in frames if f.get("hands")) / n
    pose = [f.get("pose") for f in frames]
    face = sum(1 for p in pose if p is not None and min(p[j][3] if len(p[j]) > 3 else 1 for j in (0, 2, 5)) >= 0.5) / n
    edge = 0
    for f in frames:
        for hd in f.get("hands") or []:
            xs = [q[0] for q in hd["lm"]]
            ys = [q[1] for q in hd["lm"]]
            if min(xs) < 0.01 or max(xs) > 0.99 or min(ys) < 0.01 or max(ys) > 0.99:
                edge += 1
                break
    return {"frames": len(frames), "hand_rate": round(hand, 3), "face_rate": round(face, 3),
            "edge_rate": round(edge / n, 3)}
