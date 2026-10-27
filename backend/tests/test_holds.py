"""Hold detection: still + stable-shape runs are found, moves and handshape transitions are not."""
import numpy as np

from app import holds
from tests import synth


def _clip(rng=None, noise=0.0):
    """30 fps: move (0-9), hold open (10-24), curl in place (25-32), hold curled (33-44), hand gone (45-49)."""
    frames = []
    for i in range(50):
        if i < 10:
            x, curl = 0.30 + 0.03 * i, 0.0
        else:
            x = 0.60
            curl = 0.0 if i < 25 else min((i - 24) / 8, 1.0)
        hands = [] if i >= 45 else [{"handedness": "Left", "score": 0.95,
                                     "lm": synth.hand((x, 0.55), size=0.12, curl=curl, rng=rng, noise=noise).tolist()}]
        frames.append({"t": i * 1000 / 30, "w": 640, "h": 480, "hands": hands,
                       "pose": synth.pose(wrist_r=(x, 0.55))})
    return frames


def test_finds_the_two_holds_and_skips_transitions():
    frames = _clip()
    found = holds.find_holds(frames)
    assert len(found) == 2, found
    a, b = found
    assert a.start >= 9 and a.end <= 25 and set(range(13, 23)) <= set(range(a.start, a.end + 1))
    assert b.start >= 29 and b.end == 44
    assert holds.longest(found) is a
    for h in found:
        assert all(h.start <= i <= h.end for i in holds.sample(h))
    picks = holds.sample(a, k=5, inner=0.6)
    assert len(picks) == 5 and picks[0] > a.start and picks[-1] < a.end


def test_tolerates_landmark_jitter():
    for noise in (0.0015, 0.003):
        found = holds.find_holds(_clip(rng=np.random.default_rng(0), noise=noise))
        assert len(found) == 2, noise


def test_frames_near_rejects_peaks_in_transitions():
    m = holds.motion(_clip())
    assert len(holds.frames_near(m, 18, radius=2)) == 5
    assert holds.frames_near(m, 28, radius=1) == []
    assert holds.frames_near(m, 47, radius=1) == []


def test_training_learns_only_from_hold_frames():
    from app.train_static import hold_frames
    clip = _clip()
    spans = [range(h.start, h.end + 1) for h in holds.find_holds(clip)]
    idx = hold_frames(clip)
    assert idx and all(any(i in s for s in spans) for i in idx)
    assert not any(i < 10 or 25 <= i <= 32 for i in idx)
    assert hold_frames(clip[:1]) == [0]          # a photo keeps its single frame


def test_wrist_speed_matches_segmenter_scale():
    f0, f1 = _clip()[0], _clip()[1]
    v = holds.wrist_speed(f0, f1)
    shoulder = np.hypot((synth.POSE_BASE[11][0] - synth.POSE_BASE[12][0]) * 640 / 480, 0)
    assert abs(v - 0.03 * 640 / 480 / shoulder * 30) < 1e-6
