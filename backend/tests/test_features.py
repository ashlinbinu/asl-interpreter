import numpy as np
import pytest

from app import features as fx
from tests import synth


def test_shapes():
    ex = fx.extract(synth.sign_clip(n=30))
    assert ex.feats.shape == (30, fx.F)
    seq = fx.resample(ex.feats, ex.t, ex.hand_mask)
    assert seq.shape == (fx.T, fx.F)
    assert ex.body is not None and ex.body.scale == pytest.approx(0.24 * 640 / 480, rel=1e-6)


def test_too_few_hand_frames_is_none():
    assert fx.to_sequence(synth.sign_clip(n=3)) is None
    assert fx.to_sequence([]) is None


def test_translation_and_scale_invariance():
    a = synth.sign_clip(seed=7, noise=0)
    b = []
    for f in a:  # move and shrink the whole scene (camera further away, signer off-centre)
        tf = lambda x, y: (0.1 + 0.7 * x, 0.05 + 0.7 * y)  # noqa: E731
        hands = [{**h, "lm": [[*tf(x, y), 0.7 * z] for x, y, z in h["lm"]]} for h in f["hands"]]
        pose = [[*tf(q[0], q[1]), q[2], q[3]] for q in f["pose"]]
        b.append({**f, "hands": hands, "pose": pose})
    fa, fb = fx.extract(a).feats, fx.extract(b).feats
    # Smoothing constants are in absolute units, so compare shape exactly but motion loosely.
    np.testing.assert_allclose(fa[:, 1:94], fb[:, 1:94], atol=1e-6)
    np.testing.assert_allclose(fa[:, 94:122], fb[:, 94:122], atol=0.02)


def test_left_handed_mirror_matches_right_handed():
    right = synth.sign_clip(seed=3, two_hands=True)
    left = synth.mirror_frames(right)
    np.testing.assert_allclose(fx.extract(right).feats, fx.extract(left, left_handed=True).feats, atol=1e-9)


def test_hand_order_does_not_matter_with_pose():
    a = synth.sign_clip(seed=4, two_hands=True)
    b = [{**f, "hands": list(reversed(f["hands"]))} for f in a]
    np.testing.assert_allclose(fx.extract(a).feats, fx.extract(b).feats, atol=1e-12)


def test_label_fallback_without_pose():
    # MediaPipe "Left" on an un-mirrored image is the person's right hand -> dominant slot 0.
    frames = synth.sign_clip(seed=5, with_pose=False)
    ex = fx.extract(frames)
    assert ex.hand_mask[:, 0].all() and not ex.hand_mask[:, 1].any()
    assert ex.body is None
    assert np.all(ex.feats[:, 94:128] == 0)


def test_gap_fill_short_but_not_long():
    short = fx.extract(synth.sign_clip(drop=(10, 11)))
    assert short.hand_mask[:, 0].all()
    long = fx.extract(synth.sign_clip(drop=tuple(range(10, 15))))
    assert not long.hand_mask[10:15, 0].any()


def test_location_separates_forehead_and_chin():
    chin = fx.to_sequence(synth.sign_clip(path="chin_forward", noise=0))
    head = fx.to_sequence(synth.sign_clip(path="forehead", noise=0))
    forehead_i = 100 + fx.ANCHORS.index("forehead")   # index-tip -> forehead distance
    chin_i = 100 + fx.ANCHORS.index("chin")
    assert head[:, forehead_i].mean() < chin[:, forehead_i].mean()
    assert chin[:8, chin_i].mean() < head[:8, chin_i].mean()


def test_motion_features_nonzero_only_when_moving():
    still = fx.extract(synth.sign_clip(path="still", noise=0)).feats
    moving = fx.extract(synth.sign_clip(path="chest_circle", noise=0)).feats
    assert np.abs(still[:, 122:128]).max() < 1e-9
    assert np.abs(moving[:, 122:128]).max() > 0.1


def test_static_vector_left_hand_mirrors_to_right():
    rh = {"t": 0, "w": 300, "h": 300, "pose": None,
          "hands": [{"handedness": "Left", "score": 1, "lm": synth.hand((0.5, 0.8), size=0.3, curl=0.4).tolist()}]}
    lh = {"t": 0, "w": 300, "h": 300, "pose": None,
          "hands": [{"handedness": "Right", "score": 1,
                     "lm": synth.hand((0.5, 0.8), size=0.3, curl=0.4, left=True).tolist()}]}
    vr, _ = fx.static_vectors([rh])
    vl, _ = fx.static_vectors([lh])
    assert vr.shape == (1, fx.STATIC_DIM)
    np.testing.assert_allclose(vr, vl, atol=1e-9)


def test_curl_changes_flexion_angles():
    open_v = fx.static_vectors([[synth.hand((0.5, 0.6), curl=0.0).tolist()]])[0][0]
    fist_v = fx.static_vectors([[synth.hand((0.5, 0.6), curl=1.0).tolist()]])[0][0]
    # flexion angle at index PIP: ~straight (≈1.0 after /π) when open, bent when curled
    assert open_v[63 + 4] > fist_v[63 + 4] + 0.1


def test_legacy_frames_accepted():
    legacy = [[synth.hand((0.5, 0.6)).tolist()] for _ in range(6)]
    assert fx.to_sequence(legacy).shape == (fx.T, fx.F)


def test_groups_cover_disjoint_ranges():
    seen = set()
    for idx in fx.GROUPS.values():
        s = set(idx.tolist())
        assert not (s & seen)
        seen |= s
    assert max(seen) < fx.F
