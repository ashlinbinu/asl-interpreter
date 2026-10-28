"""Landmark -> feature pipeline, implementing docs/FEATURE_SPEC.md (v1.0).

Mirrored line-for-line by frontend/src/lib/features.js; tests/test_feature_parity.py keeps them equal.
Used by the API, the dataset ingest scripts and training, so features always match.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

SPEC_VERSION = "1.0"
T = 32
F = 270
HAND = 128
GLOBAL = 256
STATIC_DIM = 93
GAP_MAX = 3
MIN_CUTOFF, BETA, D_CUTOFF = 1.5, 5.0, 1.0
VIS_MIN, WRIST_VIS_MIN = 0.5, 0.3
POSE_N = 17
MIN_HAND_FRAMES = 4

POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
FINGER_DIRS = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
TIP_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12)]
PALM = [0, 5, 9, 13, 17]
ANCHORS = ["forehead", "eye_d", "eye_nd", "nose", "mouth", "chin",
           "cheek_d", "cheek_nd", "ear_d", "ear_nd", "chest"]
FACE_POINTS = [0, 2, 5, 9, 10]
FLAG_IDX = [0, HAND, GLOBAL, GLOBAL + 1]


def _group(hand_ranges, global_ranges=()):
    idx = []
    for a, b in hand_ranges:
        idx += list(range(a, b)) + list(range(HAND + a, HAND + b))
    for a, b in global_ranges:
        idx += list(range(a, b))
    return np.array(idx, dtype=int)


# Feature index groups for per-component scoring (FEATURE_SPEC §7).
GROUPS = {
    "handshape": _group([(1, 88)]),
    "orientation": _group([(88, 94)]),
    "location": _group([(94, 122)], [(258, 266)]),
    "movement": _group([(122, 128)]),
    "two_hand": _group([], [(266, 270)]),
}


@dataclass
class Body:
    ox: float
    oy: float
    scale: float
    roll: float

    def to_body(self, xy: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.roll), math.sin(self.roll)
        dx, dy = xy[..., 0] - self.ox, xy[..., 1] - self.oy
        return np.stack([(dx * c + dy * s) / self.scale, (-dx * s + dy * c) / self.scale], axis=-1)


@dataclass
class Extracted:
    feats: np.ndarray       # (N, F) per-frame features at the original rate
    t: np.ndarray           # (N,) seconds
    hand_mask: np.ndarray   # (N, 2) slot presence after gap fill
    body: Optional[Body]


# ---------------------------------------------------------------- parsing / canonicalisation

def normalize_frame(f, i=0) -> dict:
    """Accept a RawFrame dict or a legacy bare list of hands."""
    if isinstance(f, dict):
        return f
    return {"t": i * 1000.0 / 30, "w": 640, "h": 480,
            "hands": [{"handedness": None, "lm": h} for h in (f or [])], "pose": None}


def _pose_sides(hands, pose):
    """Person side ('R'/'L') per hand by nearest pose wrist; both wrists must be visible (FEATURE_SPEC §2.2)."""
    wl, wr = pose[15, :2], pose[16, :2]
    d = [(float(np.hypot(*(h[0, :2] - wl))), float(np.hypot(*(h[0, :2] - wr)))) for h in hands]
    if len(hands) == 1:
        return ["L" if d[0][0] < d[0][1] else "R"]
    straight = d[0][1] + d[1][0]    # hand0 -> R, hand1 -> L
    crossed = d[0][0] + d[1][1]     # hand0 -> L, hand1 -> R
    return ["R", "L"] if straight <= crossed else ["L", "R"]


def _label_sides(labels, xs):
    """Fallback: MediaPipe labels assume a mirrored image, so 'Left' is the person's right hand."""
    sides = [{"Left": "R", "Right": "L"}.get(lab) for lab in labels]
    if len(sides) == 2 and (None in sides or sides[0] == sides[1]):
        return ["R", "L"] if xs[0] < xs[1] else ["L", "R"]
    if len(sides) == 1 and sides[0] is None:
        return ["R"]
    return sides


def canonicalize(frames, left_handed=False):
    """RawFrames -> canonical arrays (FEATURE_SPEC §2)."""
    n = len(frames)
    hands = np.zeros((n, 2, 21, 3))
    hmask = np.zeros((n, 2), dtype=bool)
    pose = np.zeros((n, POSE_N, 3))      # x, y, vis
    pmask = np.zeros(n, dtype=bool)
    t = np.zeros(n)
    for i, raw in enumerate(frames):
        f = normalize_frame(raw, i)
        t[i] = float(f.get("t", i * 1000.0 / 30)) / 1000.0
        w, h = float(f.get("w") or 640), float(f.get("h") or 480)
        a = w / h if h else 4 / 3
        p = None
        if f.get("pose") is not None and len(f["pose"]) >= POSE_N:
            arr = np.asarray(f["pose"], dtype=float)[:POSE_N]
            p = np.zeros((POSE_N, 3))
            p[:, 0] = arr[:, 0] * a
            p[:, 1] = arr[:, 1]
            p[:, 2] = arr[:, 3] if arr.shape[1] > 3 else 1.0
        hs, labels = [], []
        for hd in (f.get("hands") or [])[:2]:
            lm = np.asarray(hd["lm"] if isinstance(hd, dict) else hd, dtype=float).reshape(21, 3)
            hs.append(lm * np.array([a, 1.0, a]))
            labels.append(hd.get("handedness") if isinstance(hd, dict) else None)
        if p is not None and p[15, 2] >= WRIST_VIS_MIN and p[16, 2] >= WRIST_VIS_MIN and hs:
            sides = _pose_sides(hs, p)
        else:
            sides = _label_sides(labels, [hh[0, 0] for hh in hs])
        if left_handed:
            for hh in hs:
                hh[:, 0] = a - hh[:, 0]
            sides = [{"R": "L", "L": "R"}[s] for s in sides]
            if p is not None:
                p[:, 0] = a - p[:, 0]
                for x, y in POSE_SWAP:
                    p[[x, y]] = p[[y, x]]
        for hh, s in zip(hs, sides):
            k = 0 if s == "R" else 1
            hands[i, k] = hh
            hmask[i, k] = True
        if p is not None:
            pose[i] = p
            pmask[i] = True
    return hands, hmask, pose, pmask, t


# ---------------------------------------------------------------- gap fill + smoothing

def _fill_gaps(values, mask, t):
    """values: (N, ...) ; mask: (N,) bool. Linear fill of interior gaps <= GAP_MAX (in place)."""
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        # missing run i..j-1
        if i > 0 and j < n and (j - i) <= GAP_MAX:
            a, b = i - 1, j
            span = t[b] - t[a]
            for k in range(i, j):
                alpha = (t[k] - t[a]) / span if span > 0 else (k - a) / (b - a)
                values[k] = (1 - alpha) * values[a] + alpha * values[b]
                mask[k] = True
        i = j
    return values, mask


def _alpha(dt, fc):
    tau = 1.0 / (2 * math.pi * fc)
    return 1.0 / (1.0 + tau / dt)


def _one_euro(values, mask, t):
    """Causal One-Euro filter per scalar; values (N, D)."""
    out = values.copy()
    n = len(mask)
    x_hat = np.zeros(values.shape[1])
    dx_hat = np.zeros(values.shape[1])
    for i in range(n):
        if not mask[i]:
            continue
        x = values[i]
        if i == 0 or not mask[i - 1]:
            x_hat = x.copy()
            dx_hat = np.zeros_like(x)
        else:
            dt = t[i] - t[i - 1]
            if dt <= 0:
                dt = 1.0 / 30
            dx = (x - x_hat) / dt
            ad = _alpha(dt, D_CUTOFF)
            dx_hat = ad * dx + (1 - ad) * dx_hat
            fc = MIN_CUTOFF + BETA * np.abs(dx_hat)
            tau = 1.0 / (2 * math.pi * fc)
            a = 1.0 / (1.0 + tau / dt)
            x_hat = a * x + (1 - a) * x_hat
        out[i] = x_hat
    return out


# ---------------------------------------------------------------- body frame + anchors

def _median(v):
    return float(np.median(np.asarray(v, dtype=float)))


def body_frame(pose, pmask) -> Optional[Body]:
    ok = pmask & (pose[:, 11, 2] >= VIS_MIN) & (pose[:, 12, 2] >= VIS_MIN)
    if not ok.any():
        return None
    L, R = pose[ok, 11, :2], pose[ok, 12, :2]
    o = (L + R) / 2
    scale = np.hypot(*(L - R).T)
    roll = np.arctan2(L[:, 1] - R[:, 1], L[:, 0] - R[:, 0])
    b = Body(_median(o[:, 0]), _median(o[:, 1]), _median(scale), _median(roll))
    return b if b.scale >= 1e-6 else None


def _anchors(p, body: Body):
    """Face anchors 0..9 in body coords (FEATURE_SPEC §6); p = (17, 3) pose."""
    P = p[:, :2]
    eyes = (P[2] + P[5]) / 2
    mouth = (P[9] + P[10]) / 2
    up = eyes - mouth
    pts = np.stack([eyes + 0.8 * up, P[5], P[2], P[0], mouth, mouth - 0.7 * up,
                    (P[10] + P[8]) / 2, (P[9] + P[7]) / 2, P[8], P[7]])
    return body.to_body(pts)


# ---------------------------------------------------------------- per-frame features

def _rot(v, roll):
    c, s = math.cos(roll), math.sin(roll)
    out = v.copy()
    out[..., 0] = v[..., 0] * c + v[..., 1] * s
    out[..., 1] = -v[..., 0] * s + v[..., 1] * c
    return out


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _angle(u, v):
    d = float(np.dot(u, v)) / max(float(np.linalg.norm(u)) * float(np.linalg.norm(v)), 1e-12)
    return math.acos(max(-1.0, min(1.0, d))) / math.pi


def _hand_block(p, chir, body, anchors, chest):
    """One hand's 128 values minus velocity (filled later)."""
    out = np.zeros(HAND)
    out[0] = 1.0
    roll = body.roll if body else 0.0
    r = _rot(p - p[0], roll)
    s_h = max(float(np.linalg.norm(r[9])), 1e-6)
    out[1:64] = (r / s_h).reshape(-1)
    k = 64
    for ch in CHAINS:
        for j in (1, 2, 3):
            out[k] = _angle(r[ch[j - 1]] - r[ch[j]], r[ch[j + 1]] - r[ch[j]])
            k += 1
    dirs = [r[b] - r[a] for a, b in FINGER_DIRS]
    for j in range(4):
        out[79 + j] = _angle(dirs[j], dirs[j + 1])
    for j, (a, b) in enumerate(TIP_PAIRS):
        out[83 + j] = float(np.linalg.norm(r[a] - r[b])) / s_h
    out[88:91] = chir * _unit(np.cross(r[5], r[17]))
    out[91:94] = _unit(r[9])
    if body is not None:
        locs = body.to_body(np.stack([p[0, :2], p[8, :2], p[PALM, :2].mean(axis=0)]))
        out[94:100] = locs.reshape(-1)
        for qi, q in enumerate((locs[1], locs[2])):
            base = 100 + qi * 11
            if anchors is not None:
                out[base:base + 10] = np.hypot(*(anchors - q).T)
            out[base + 10] = float(np.hypot(*(chest - q)))
    return out


def extract(frames, left_handed=False) -> Extracted:
    """RawFrames -> per-frame (N, F) features (FEATURE_SPEC §2-7)."""
    hands, hmask, pose, pmask, t = canonicalize(frames, left_handed)
    n = len(t)
    for k in range(2):
        flat = hands[:, k].reshape(n, -1)
        flat, hmask[:, k] = _fill_gaps(flat, hmask[:, k].copy(), t)
        hands[:, k] = _one_euro(flat, hmask[:, k], t).reshape(n, 21, 3)
    pflat = pose.reshape(n, -1)
    pflat, pmask = _fill_gaps(pflat, pmask.copy(), t)
    pose = pflat.reshape(n, POSE_N, 3)
    xy = pose[:, :, :2].reshape(n, -1)
    pose[:, :, :2] = _one_euro(xy, pmask, t).reshape(n, POSE_N, 2)

    body = body_frame(pose, pmask)
    chest = np.array([0.0, 0.6])
    feats = np.zeros((n, F))
    for i in range(n):
        anchors = None
        face = False
        if body is not None and pmask[i] and all(pose[i, j, 2] >= VIS_MIN for j in FACE_POINTS):
            anchors = _anchors(pose[i], body)
            face = True
        for k in range(2):
            if hmask[i, k]:
                feats[i, k * HAND:(k + 1) * HAND] = _hand_block(hands[i, k], 1.0 if k == 0 else -1.0,
                                                               body, anchors, chest)
        g = feats[i, GLOBAL:]
        body_i = body is not None and pmask[i] and pose[i, 11, 2] >= VIS_MIN and pose[i, 12, 2] >= VIS_MIN
        g[1] = 1.0 if face else 0.0
        if body_i:
            g[0] = 1.0
            g[2:4] = body.to_body(pose[i, 12, :2])
            g[4:6] = body.to_body(pose[i, 11, :2])
            if pose[i, 14, 2] >= VIS_MIN:
                g[6:8] = body.to_body(pose[i, 14, :2])
            if pose[i, 13, 2] >= VIS_MIN:
                g[8:10] = body.to_body(pose[i, 13, :2])
        if body is not None and hmask[i, 0] and hmask[i, 1]:
            w0, w1 = feats[i, 94:96], feats[i, HAND + 94:HAND + 96]
            d = w0 - w1
            g[10:12] = d
            g[12] = float(np.hypot(*d))
            g[13] = float(np.hypot(*(feats[i, 96:98] - feats[i, HAND + 98:HAND + 100])))
    if body is not None:
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            if dt <= 0:
                dt = 1.0 / 30
            for k in range(2):
                if hmask[i, k] and hmask[i - 1, k]:
                    o = k * HAND
                    feats[i, o + 122:o + 128] = (feats[i, o + 94:o + 100] - feats[i - 1, o + 94:o + 100]) / dt
    return Extracted(feats, t, hmask, body)


# ---------------------------------------------------------------- resampling + static vectors

def resample(feats, t, hand_mask, length=T) -> Optional[np.ndarray]:
    """Per-frame features -> (length, F) over the hand-present span (FEATURE_SPEC §8)."""
    any_hand = hand_mask.any(axis=1)
    if int(any_hand.sum()) < MIN_HAND_FRAMES:
        return None
    idx = np.flatnonzero(any_hand)
    a, b = int(idx[0]), int(idx[-1])
    seg, ts = feats[a:b + 1], t[a:b + 1]
    n = len(seg)
    span = ts[-1] - ts[0]
    u = (ts - ts[0]) / span if span > 0 else np.arange(n) / max(n - 1, 1)
    out = np.zeros((length, feats.shape[1]))
    for k in range(length):
        uk = k / (length - 1)
        i = int(np.searchsorted(u, uk, side="right")) - 1
        i = min(max(i, 0), n - 2) if n > 1 else 0
        if n == 1:
            out[k] = seg[0]
            continue
        du = u[i + 1] - u[i]
        alpha = (uk - u[i]) / du if du > 0 else 0.0
        alpha = min(max(alpha, 0.0), 1.0)
        if all(seg[i, j] == seg[i + 1, j] for j in FLAG_IDX):
            out[k] = (1 - alpha) * seg[i] + alpha * seg[i + 1]
        else:
            out[k] = seg[i] if alpha < 0.5 else seg[i + 1]
    return out


def to_sequence(frames, left_handed=False) -> Optional[np.ndarray]:
    """RawFrames -> (T, F) classifier input, or None if too few hand frames."""
    if not frames:
        return None
    ex = extract(frames, left_handed)
    return resample(ex.feats, ex.t, ex.hand_mask)


def static_vector(row) -> Optional[np.ndarray]:
    """One per-frame vector -> 93-dim right-chirality hand-shape vector (FEATURE_SPEC §9)."""
    if row[0] == 1.0:
        return row[1:1 + STATIC_DIM].copy()
    if row[HAND] == 1.0:
        v = row[HAND + 1:HAND + 1 + STATIC_DIM].copy()
        v[0:63:3] *= -1          # shape x
        v[87] *= -1              # normal x  (block offset 88 -> vector index 87)
        v[90] *= -1              # direction x (block offset 91 -> vector index 90)
        return v
    return None


def static_vectors(frames, left_handed=False):
    """RawFrames -> (M, 93) static vectors for frames that contain a hand, plus their frame indices."""
    if not frames:
        return np.zeros((0, STATIC_DIM)), np.zeros(0, dtype=int)
    ex = extract(frames, left_handed)
    rows, idx = [], []
    for i, row in enumerate(ex.feats):
        v = static_vector(row)
        if v is not None:
            rows.append(v)
            idx.append(i)
    return (np.array(rows) if rows else np.zeros((0, STATIC_DIM))), np.array(idx, dtype=int)
