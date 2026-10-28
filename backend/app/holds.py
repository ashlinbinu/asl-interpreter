"""Hold detection in fingerspelling video (docs/PLAN.md §7.2): frames where the hand is still and its shape is stable.

Turns letter videos into static training samples. J and Z are motion letters and use whole segments instead.
Wrist speed is the browser segmenter's measure (frontend/src/lib/segmenter.js), so offline holds and live static
capture agree on what "still" means. Shape speed is the change of the 93-dim static vector (FEATURE_SPEC §9) per
second.
"""
from dataclasses import dataclass

import numpy as np

from . import features as fx

WRIST_V = 0.45       # body units / s (segmenter.js stillV)
SHAPE_V = 4.0        # static-vector units / s (jitter ≈1-2, a handshape change ≈15 in tests); refit on real holds
MIN_HOLD_MS = 120


def _scale(f) -> float:
    a = (f.get("w") or 640) / (f.get("h") or 480)
    p = f.get("pose")
    if p is not None and len(p) > 12 and p[11][3] >= 0.5 and p[12][3] >= 0.5:
        return float(np.hypot((p[11][0] - p[12][0]) * a, p[11][1] - p[12][1]))
    h = (f.get("hands") or [{}])[0].get("lm")
    return 2.5 * float(np.hypot((h[9][0] - h[0][0]) * a, h[9][1] - h[0][1])) if h else 0.0


def wrist_speed(prev, cur) -> float:
    """Max wrist speed between two RawFrames (hands matched by nearest wrist), as in segmenter.js."""
    if not prev or not prev.get("hands") or not cur.get("hands"):
        return 0.0
    a = (cur.get("w") or 640) / (cur.get("h") or 480)
    dt = max((cur["t"] - prev["t"]) / 1000, 1 / 120)
    s = _scale(cur) or 1.0
    best = 0.0
    for h in cur["hands"]:
        d = min(np.hypot((h["lm"][0][0] - g["lm"][0][0]) * a, h["lm"][0][1] - g["lm"][0][1]) for g in prev["hands"])
        best = max(best, d / s / dt)
    return float(best)


@dataclass
class Hold:
    start: int          # first frame index (inclusive)
    end: int            # last frame index (inclusive)
    t0: float           # ms
    t1: float

    @property
    def ms(self) -> float:
        return self.t1 - self.t0


def motion(frames, left_handed=False, half_window_ms=50.0):
    """Per frame: t (ms), hand present, wrist speed, shape speed (inf where undefined), static vectors (nan rows).
    Speeds are centred differences over about ±half_window_ms (±2 frames at 30 fps), which averages out landmark
    jitter; at the edge of the visible span the window shrinks to the frames that have a hand."""
    n = len(frames)
    t = np.array([float(f.get("t", i * 1000 / 30)) for i, f in enumerate(frames)])
    vecs = np.full((n, fx.STATIC_DIM), np.nan)
    if n:
        for i, row in enumerate(fx.extract(frames, left_handed).feats):
            v = fx.static_vector(row)
            if v is not None:
                vecs[i] = v
    has = ~np.isnan(vecs[:, 0])
    step = float(np.median(np.diff(t))) if n > 1 else 1000 / 30
    w = max(1, int(round(half_window_ms / max(step, 1e-3))))
    wv, sv = np.full(n, np.inf), np.full(n, np.inf)
    for i in np.flatnonzero(has):
        a = next(j for j in range(max(i - w, 0), i + 1) if has[j])
        b = next(j for j in range(min(i + w, n - 1), i - 1, -1) if has[j])
        if a == b:
            continue
        wv[i] = wrist_speed(frames[a], frames[b])
        sv[i] = float(np.linalg.norm(vecs[b] - vecs[a])) / max((t[b] - t[a]) / 1000, 1 / 120)
    return t, has, wv, sv, vecs


def find_holds(frames, left_handed=False, wrist_v=WRIST_V, shape_v=SHAPE_V, min_ms=MIN_HOLD_MS) -> list[Hold]:
    """Maximal runs of frames with a hand, wrist speed < wrist_v and shape speed < shape_v, lasting >= min_ms."""
    t, has, wv, sv, _ = motion(frames, left_handed)
    still = has & (wv < wrist_v) & (sv < shape_v)
    holds, i, n = [], 0, len(frames)
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and still[j + 1]:
            j += 1
        if t[j] - t[i] >= min_ms:
            holds.append(Hold(i, j, float(t[i]), float(t[j])))
        i = j + 1
    return holds


def sample(hold: Hold, k=5, inner=0.6) -> list[int]:
    """Up to k frame indices spread over the middle `inner` fraction of a hold (avoids the transitions)."""
    n = hold.end - hold.start + 1
    pad = (1 - inner) / 2 * (n - 1)
    lo, hi = hold.start + pad, hold.end - pad
    return sorted({int(round(x)) for x in np.linspace(lo, hi, min(k, n))})


def inner(hold: Hold, fraction=0.6) -> list[int]:
    """All frame indices in the middle `fraction` of a hold."""
    pad = int((1 - fraction) / 2 * (hold.end - hold.start + 1))
    return list(range(hold.start + pad, hold.end - pad + 1))


def longest(holds: list[Hold]) -> Hold | None:
    return max(holds, key=lambda h: h.ms) if holds else None


def frames_near(m, peak: int, radius=2, wrist_v=WRIST_V, shape_v=SHAPE_V) -> list[int]:
    """Frames within `radius` of an annotated peak that pass the stillness test (rejects transitions).
    m = motion(frames), computed once per video."""
    _, has, wv, sv, _ = m
    lo, hi = max(peak - radius, 0), min(peak + radius, len(has) - 1)
    return [i for i in range(lo, hi + 1) if has[i] and wv[i] < wrist_v and sv[i] < shape_v]
