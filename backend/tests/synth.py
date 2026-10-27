"""Deterministic RawFrame generators for TESTS ONLY (geometry checks, parity, API plumbing).

These are not sign references and are never written to the database.
"""
import numpy as np

# Open right hand, fingers up, palm toward the camera; units = wrist->middle-MCP length.
_OPEN = np.array([
    [0.00, 0.00, 0.00],
    [-0.35, -0.20, -0.02], [-0.60, -0.45, -0.04], [-0.78, -0.68, -0.05], [-0.92, -0.88, -0.06],
    [-0.30, -0.95, -0.02], [-0.33, -1.35, -0.03], [-0.35, -1.60, -0.04], [-0.37, -1.82, -0.05],
    [0.00, -1.00, -0.01], [0.00, -1.45, -0.02], [0.00, -1.72, -0.03], [0.00, -1.95, -0.04],
    [0.27, -0.93, -0.01], [0.30, -1.32, -0.02], [0.32, -1.57, -0.03], [0.33, -1.78, -0.04],
    [0.50, -0.82, 0.00], [0.56, -1.12, -0.01], [0.60, -1.32, -0.02], [0.63, -1.50, -0.03],
])
_FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

POSE_BASE = {
    0: (0.50, 0.33), 1: (0.52, 0.30), 2: (0.53, 0.30), 3: (0.54, 0.30), 4: (0.48, 0.30), 5: (0.47, 0.30),
    6: (0.46, 0.30), 7: (0.57, 0.32), 8: (0.43, 0.32), 9: (0.52, 0.40), 10: (0.48, 0.40),
    11: (0.62, 0.62), 12: (0.38, 0.62), 13: (0.68, 0.85), 14: (0.32, 0.85), 15: (0.62, 0.95), 16: (0.38, 0.95),
}
POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def hand(wrist_xy, size=0.09, curl=0.0, rot=0.0, left=False, rng=None, noise=0.0):
    """21x3 normalised landmarks. curl in [0, 1] folds the four fingers toward the palm."""
    p = _OPEN.copy()
    theta = curl * np.radians(80)          # bend at each of MCP, PIP, DIP toward the palm (-z)
    for chain in _FINGERS:
        pts = _OPEN[list(chain)]
        lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        d0 = (pts[1] - pts[0]) / lengths[0]
        nrm = np.array([0.0, 0.0, -1.0])
        cur = pts[0].copy()
        for k, (j, L) in enumerate(zip(chain[1:], lengths), start=1):
            cur = cur + L * (np.cos(k * theta) * d0 + np.sin(k * theta) * nrm)
            p[j] = cur
    c, s = np.cos(rot), np.sin(rot)
    p[:, :2] = p[:, :2] @ np.array([[c, s], [-s, c]])
    if left:
        p[:, 0] *= -1
    p = p * size
    p[:, 0] += wrist_xy[0]
    p[:, 1] += wrist_xy[1]
    if rng is not None and noise:
        p = p + rng.normal(0, noise, p.shape)
    return p


def pose(wrist_r=None, wrist_l=None, vis=0.99, shift=(0.0, 0.0)):
    pts = []
    for i in range(17):
        x, y = POSE_BASE[i]
        if i == 16 and wrist_r is not None:
            x, y = wrist_r
        if i == 15 and wrist_l is not None:
            x, y = wrist_l
        pts.append([x + shift[0], y + shift[1], 0.0, vis])
    return pts


def sign_clip(n=30, fps=30.0, path="chin_forward", two_hands=False, curl=0.0, seed=0,
              drop=(), noise=0.002, w=640, h=480, with_pose=True):
    """A right-handed 'sign': dominant hand moves along `path`; optional static non-dominant hand."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        u = i / max(n - 1, 1)
        if path == "chin_forward":
            wr = (0.47 + 0.02 * u, 0.50 + 0.12 * u)
        elif path == "forehead":
            wr = (0.47, 0.30 + 0.01 * np.sin(6 * u))
        elif path == "chest_circle":
            wr = (0.47 + 0.05 * np.cos(2 * np.pi * u), 0.75 + 0.05 * np.sin(2 * np.pi * u))
        else:
            wr = (0.40, 0.80)
        hands = [] if i in drop else [{"handedness": "Left", "score": 0.95,
                                       "lm": hand(wr, curl=curl, rng=rng, noise=noise).tolist()}]
        wl = (0.60, 0.85)
        if two_hands and i not in drop:
            hands.append({"handedness": "Right", "score": 0.93,
                          "lm": hand(wl, curl=0.2, left=True, rng=rng, noise=noise).tolist()})
        frames.append({"t": round(i * 1000.0 / fps, 3), "w": w, "h": h, "hands": hands,
                       "pose": pose(wr, wl) if with_pose else None})
    return frames


def mirror_frames(frames):
    """Mirror raw frames horizontally (what a left-handed signer doing the same sign looks like)."""
    out = []
    for f in frames:
        hands = [{"handedness": {"Left": "Right", "Right": "Left"}.get(hd["handedness"]), "score": hd["score"],
                  "lm": [[1 - x, y, z] for x, y, z in hd["lm"]]} for hd in f["hands"]]
        p = None
        if f["pose"] is not None:
            p = [[1 - q[0], q[1], q[2], q[3]] for q in f["pose"]]
            for a, b in POSE_SWAP:
                p[a], p[b] = p[b], p[a]
        out.append({**f, "hands": hands, "pose": p})
    return out


def parity_cases():
    """A mix of situations both feature implementations must agree on."""
    return [
        {"frames": sign_clip(seed=1), "left_handed": False},
        {"frames": sign_clip(seed=2, two_hands=True, path="chest_circle", drop=(5, 6, 14, 15, 16, 17, 18)),
         "left_handed": False},
        {"frames": mirror_frames(sign_clip(seed=3, path="forehead")), "left_handed": True},
        {"frames": sign_clip(seed=4, with_pose=False, w=1280, h=720), "left_handed": False},
        {"frames": sign_clip(seed=5, n=3), "left_handed": False},
        {"frames": [[hand((0.5, 0.6), curl=c).tolist()] for c in np.linspace(0, 1, 8)], "left_handed": False},
        {"frames": [{"t": 0, "w": 300, "h": 300, "hands": [{"handedness": "Right", "score": 0.9,
                                                           "lm": hand((0.5, 0.8), size=0.3, left=True).tolist()}],
                     "pose": None}], "left_handed": False},
    ]
