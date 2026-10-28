"""Signs without a trained model are scored by DTW against their own clips; numba DTW equals the Python loop."""
import json

import numpy as np

from tests import synth

LEARNER = "learner-clip-1"


def test_numba_dtw_matches_python_loop():
    from app import word_model as w
    rng = np.random.default_rng(3)
    for n, m in ((32, 32), (20, 32), (32, 25)):
        a, b = rng.normal(size=(n, 12)), rng.normal(size=(m, 12))
        d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
        assert np.allclose(w._dtw_cost(d, w.BAND), w._dtw_cost_py(d, w.BAND))
    path, cost = w.dtw_path(a, a)
    assert cost == 0 and path[0] == (0, 0) and path[-1] == (len(a) - 1, len(a) - 1)


def test_custom_sign_scored_against_own_recording(client):
    ref = synth.sign_clip(seed=21, path="chin_forward")
    sid = client.post("/api/signs", json={"label": "name sign test", "frames": ref, "learner_id": LEARNER}).json()["id"]
    score = lambda frames: client.post("/api/score", json={"sign_id": sid, "frames": frames, "learner_id": LEARNER})
    same = score(synth.sign_clip(seed=22, path="chin_forward"))
    other = score(synth.sign_clip(seed=23, path="forehead", curl=1.0))
    assert same.status_code == 200 and other.status_code == 200, (same.text, other.text)
    s, o = same.json(), other.json()
    assert s["score"] > o["score"] and s["score"] >= 70
    assert "own_clip_dtw" in s["method"] and "unverified" in s["method"] and s["predicted"] is None
    assert set(s["components"]) == {"handshape", "orientation", "location", "movement", "two_hand"}
    assert "highlight" in s and s["hints"]
    p = client.get("/api/progress", params={"learner_id": LEARNER}).json()
    assert next(x for x in p["signs"] if x["sign_id"] == sid)["attempts"] == 2


def test_highlight_points_at_the_joints_that_differ():
    from app import features as fx
    from app.word_model import highlight_from
    a = np.zeros((32, fx.F))
    a[:, 0] = 1.0                                            # dominant hand present
    a[:, 1:64] = np.tile(np.arange(63) / 63.0, (32, 1))      # 21 joints x 3
    b = a.copy()
    for j, shift in ((8, 0.6), (12, 0.4), (16, 0.2), (20, 0.05)):
        b[:, 1 + 3 * j] += shift                             # move fingertips by different amounts
    h = highlight_from(a, b, [(i, i) for i in range(32)])
    assert h == {"slot": 0, "joints": [8, 12, 16], "component": "handshape"}
    assert highlight_from(a, a, [(i, i) for i in range(32)])["slot"] == 0
    c = b.copy()
    c[:, 0] = 0.0                                            # hand missing in one sequence: nothing to compare
    assert highlight_from(a, c, [(i, i) for i in range(32)]) is None


def test_anchors_and_calibration_are_monotone():
    from app.clip_scoring import ANCHORS, anchored, calibrated
    from app.word_model import COMPONENTS
    for g, (pos, neg) in ANCHORS.items():
        assert anchored(g, pos) == 75.0 and anchored(g, neg) == 25.0 and anchored(g, 0.0) > 75.0
    for combine, coef in (("all_components", [0.0, 2.6, 2.3, 0.0, 0.6]), ("mean_of_components", [5.0])):
        # Perfect match: every input is 1 (all five, or their mean), so the logit is sum(coef) + intercept.
        cal = {"neg_quantiles": {g: np.linspace(0.2, 2.0, 101).tolist() for g in COMPONENTS},
               "combine": combine, "coef": coef, "intercept": -3.7,
               "perfect_probability": 1 / (1 + np.exp(-(sum(coef) - 3.7)))}
        totals = [calibrated(cal, {g: d for g in COMPONENTS})[1] for d in (0.1, 0.5, 1.0, 1.5, 2.5)]
        assert totals == sorted(totals, reverse=True)       # farther from the recording never scores higher
        assert totals[0] == 100.0                           # a perfect match scores 100
        pct, _ = calibrated(cal, {g: 0.1 for g in COMPONENTS})
        assert all(v == 100.0 for v in pct.values())


def _letter_id(client, label):
    return next(s["id"] for s in client.get("/api/signs", params={"category": "alphabet"}).json()
                if s["label"] == label)


def test_custom_sign_without_recording_gives_a_friendly_message(client):
    from app.db import ReferenceClip, Session
    sid = client.post("/api/signs", json={"label": "no take test", "frames": synth.sign_clip(seed=2),
                                          "learner_id": LEARNER}).json()["id"]
    with Session() as s:                      # like the old custom signs that lost their recording
        s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid).delete()
        s.commit()
    from app.clip_scoring import clip_scorer
    clip_scorer.invalidate(sid)
    r = client.post("/api/score", json={"sign_id": sid, "frames": synth.sign_clip(seed=1)})
    assert r.status_code == 409
    msg = r.json()["detail"]
    assert "no recording of 'no take test'" in msg and "Add a sign" in msg and "P2" not in msg


def test_motion_letters_are_not_scored_against_own_recordings(client):
    jid = _letter_id(client, "J")
    take = synth.sign_clip(seed=31, path="chest_circle")
    up = client.post("/api/recordings", data={"meta": json.dumps({"sign_id": jid, "signer_id": "me_01"})},
                     files={"landmarks": ("l.json", json.dumps(take), "application/json")})
    assert up.status_code == 200, up.text
    r = client.post("/api/score", json={"sign_id": jid, "frames": synth.sign_clip(seed=32, path="chest_circle")})
    # J and Z are scored only against the letter reference videos (tests/test_motion_letters.py), never by DTW
    # against recordings; this test database has none, so the answer says what is missing.
    assert r.status_code == 409 and "ingest_commons_alphabet" in r.json()["detail"]
