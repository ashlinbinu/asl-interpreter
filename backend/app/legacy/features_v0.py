# Legacy v0 features (hands only, 126 dims). Dev fallback for synthetic templates only.
"""Turns raw MediaPipe landmarks into fixed-size, position/scale-invariant sequences.
Used by BOTH the API and the dataset ingest script so features always match."""
import numpy as np

T = 32          # frames per sequence after resampling
DIM = 2 * 21 * 3  # two hands x 21 landmarks x (x, y, z)


def frame_vec(hands):
    """hands: list of up to 2 hands, each 21 x [x, y, z]. Ordered left-to-right in the image."""
    out = np.zeros((2, 21, 3))
    for i, h in enumerate(sorted(hands, key=lambda h: h[0][0])[:2]):
        h = np.asarray(h, dtype=float)
        h = h - h[0]                                  # wrist at origin
        scale = np.linalg.norm(h, axis=1).max() or 1  # hand size = 1
        out[i] = h / scale
    return out.reshape(-1)


def to_sequence(frames):
    """Raw frames -> (T, DIM) array, or None if no hand was seen."""
    vecs = [frame_vec(f) for f in frames if f]  # drop frames with no hands
    if len(vecs) < 4:
        return None
    v = np.array(vecs)
    old, new = np.linspace(0, 1, len(v)), np.linspace(0, 1, T)
    return np.stack([np.interp(new, old, v[:, j]) for j in range(DIM)], axis=1)
