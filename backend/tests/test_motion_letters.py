"""J and Z: scored on handshape + fingertip path against a reference signer's letters (synthetic stand-ins)."""
import numpy as np
import pytest

from tests import synth

FINGER_JOINTS = {0: list(range(5, 9)), 1: list(range(9, 13)), 2: list(range(13, 17)), 3: list(range(17, 21))}
SOURCE, SIGNER = "test_ref_video", "ref"


def shape(wrist, only=None, curl=1.0):
    """Hand at `wrist` with all fingers curled by `curl`, except finger `only` (0 index .. 3 little) held open."""
    lm = synth.hand(wrist, size=0.1, curl=curl)
    if only is not None:
        lm[FINGER_JOINTS[only]] = synth.hand(wrist, size=0.1, curl=0.0)[FINGER_JOINTS[only]]
    return lm


def clip(path, only=None, curl=1.0, n=30, seed=0, mirror=False):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        lm = shape(path(i / (n - 1)), only, curl) + rng.normal(0, 0.001, (21, 3))
        if mirror:
            lm[:, 0] = 1 - lm[:, 0]
        frames.append({"t": i * 33.3, "w": 640, "h": 480, "pose": None,
                       "hands": [{"handedness": "Right" if mirror else "Left", "score": 0.95, "lm": lm.tolist()}]})
    return frames


def j_path(u):        # down, then hook to the side
    return 0.5 - 0.08 * max(0.0, u - 0.6) / 0.4, 0.42 + 0.12 * min(u, 0.7) / 0.7


def z_path(u):        # across, diagonally down, across
    pts = np.array([(0.42, 0.40), (0.58, 0.40), (0.42, 0.55), (0.58, 0.55)])
    k = min(int(u * 3), 2)
    return tuple(pts[k] + (pts[k + 1] - pts[k]) * (u * 3 - k))


def drift(x0, y0):
    return lambda u: (x0 + 0.02 * u, y0 + 0.01 * u)


STATIC = {"A": (None, 1.0), "B": (None, 0.0), "C": (None, 0.5), "D": (0, 1.0), "E": (None, 0.8), "F": (None, 0.3),
          "G": (0, 1.0), "H": (1, 1.0), "I": (3, 1.0), "K": (1, 0.6), "L": (0, 1.0), "O": (None, 0.6), "Y": (3, 1.0)}


@pytest.fixture(scope="module")
def references(client, tmp_root):
    from app.clipstore import save_clip
    from app.db import ReferenceClip, ReferenceVideo, Session, Sign
    from app.motion_letters import motion_scorer
    from app.registry import registry
    clips = {L: clip(drift(0.45 + 0.01 * k, 0.45), only, curl, seed=k) for k, (L, (only, curl)) in enumerate(STATIC.items())}
    clips["J"], clips["Z"] = clip(j_path, only=3, seed=50), clip(z_path, only=0, seed=51)
    with Session() as s:
        ids = {r.label: r.id for r in s.query(Sign).filter(Sign.category == "alphabet")}
        for L, frames in clips.items():
            rel = f"landmarks/{SOURCE}/{SIGNER}/{L}.npz"
            save_clip(tmp_root / "store" / rel, frames, {"label": L})
            c = ReferenceClip(sign_id=ids[L], source=SOURCE, signer_id=SIGNER, landmarks_path=rel, num_frames=len(frames))
            s.add(c)
            s.flush()
            s.add(ReferenceVideo(sign_id=ids[L], clip_id=c.id, path=f"videos/{ids[L]}/{SOURCE}.mp4"))
        s.commit()
    motion_scorer.invalidate()
    registry.load()
    yield ids
    with Session() as s:
        s.query(ReferenceVideo).filter(ReferenceVideo.path.like(f"%/{SOURCE}.mp4")).delete(synchronize_session=False)
        s.query(ReferenceClip).filter(ReferenceClip.source == SOURCE).delete(synchronize_session=False)
        s.commit()
    motion_scorer.invalidate()
    registry.load()


def test_right_letter_scores_high_and_near_misses_low(references):
    from app.motion_letters import motion_scorer as ms
    assert ms.score(clip(j_path, only=3, n=40, seed=7), "J")["score"] >= 70          # slower, different noise
    assert ms.score(clip(z_path, only=0, n=24, seed=8), "Z")["score"] >= 70          # faster
    still_i = ms.score(clip(drift(0.5, 0.45), only=3, seed=9), "J")                  # I: right hand, no movement
    assert still_i["score"] < 40 and still_i["components"]["handshape"] >= 80
    assert any("Move more" in h for h in still_i["hints"])
    assert ms.score(clip(drift(0.5, 0.45), only=0, seed=10), "Z")["score"] < 40      # D/G/L-like: still index
    assert ms.score(clip(z_path, only=0, seed=11), "J")["score"] < 40                # Z is not J
    wrong_hand = ms.score(clip(j_path, only=0, seed=12), "J")                        # J's path with the index finger
    assert wrong_hand["score"] < 40 and wrong_hand["highlight"]["joints"]
    for letter in "JZ":
        assert max(ms.leave_one_out(letter).values()) < 50


def test_left_handed_attempt(references):
    from app.motion_letters import motion_scorer as ms
    assert ms.score(clip(j_path, only=3, seed=13, mirror=True), "J", left_handed=True)["score"] >= 70


def test_api_scores_j(client, references):
    r = client.post("/api/score", json={"sign_id": references["J"], "frames": clip(j_path, only=3, seed=14)})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["method"].startswith("motion_letter") and "unverified" in d["method"]
    assert set(d["components"]) == {"handshape", "movement"} and d["score"] >= 70
