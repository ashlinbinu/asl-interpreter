"""Custom signs: a take must be long enough, can be recorded again, and its browser video becomes the reference."""
import json

from tests import synth

LEARNER = "learner-rerecord-1"


def _record(client, sid, frames, video=b"\x1aE\xdf\xa3fake-webm"):
    files = {"landmarks": ("l.json", json.dumps(frames), "application/json")}
    if video is not None:
        files["video"] = ("take.webm", video, "video/webm")
    return client.post(f"/api/signs/{sid}/recording", data={"meta": json.dumps({"learner_id": LEARNER})}, files=files)


def test_too_short_custom_take_is_rejected(client):
    r = client.post("/api/signs", json={"label": "short take", "frames": synth.sign_clip(n=6), "learner_id": LEARNER})
    assert r.status_code == 422 and "too short" in r.json()["detail"]


def test_record_again_replaces_take_and_publishes_video(client, tmp_root):
    from app.db import ReferenceClip, Session
    sid = client.post("/api/signs", json={"label": "re-record test", "frames": synth.sign_clip(seed=41),
                                          "learner_id": LEARNER}).json()["id"]
    with Session() as s:
        first = s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid).one().landmarks_path
    r = _record(client, sid, synth.sign_clip(seed=42, n=40))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reference_clips"] == 1 and d["has_video"] is True
    assert d["reference_video_url"].startswith(f"/media/videos/{sid}/user_") and d["attribution"] == "Your recording"
    assert not (tmp_root / "store" / first).exists()                        # the old take's files are gone
    v = client.get(d["reference_video_url"])
    assert v.status_code == 200 and v.content.startswith(b"\x1aE\xdf\xa3")
    detail = client.get(f"/api/signs/{sid}").json()
    assert len(detail["reference_landmarks"]["frames"]) == 40               # the new take is the reference
    s = client.post("/api/score", json={"sign_id": sid, "frames": synth.sign_clip(seed=43, n=40)})
    assert s.status_code == 200 and "own_clip_dtw" in s.json()["method"]
    # Recording again without a video keeps the published video; deleting the sign removes it.
    assert _record(client, sid, synth.sign_clip(seed=44), video=None).json()["has_video"] is True
    media = tmp_root / "store" / "media" / d["reference_video_url"].removeprefix("/media/")
    assert media.is_file()
    assert client.delete(f"/api/signs/{sid}").status_code == 200
    assert not media.exists()


def test_record_again_only_for_custom_signs(client):
    letter = next(s for s in client.get("/api/signs", params={"category": "alphabet"}).json() if s["label"] == "B")
    assert _record(client, letter["id"], synth.sign_clip(seed=45)).status_code == 403
    assert _record(client, 999999, synth.sign_clip(seed=45)).status_code == 404
    sid = client.post("/api/signs", json={"label": "re-record short", "frames": synth.sign_clip(seed=46),
                                          "learner_id": LEARNER}).json()["id"]
    r = _record(client, sid, synth.sign_clip(seed=47, n=5))
    assert r.status_code == 422 and "too short" in r.json()["detail"]
