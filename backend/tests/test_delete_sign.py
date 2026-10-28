"""DELETE /api/signs/{id}: only custom ("user") signs; files, clips and cache entries go, attempts stay."""
import json

from tests import synth

LEARNER = "learner-delete-1"


def _user_sign(client, label):
    r = client.post("/api/signs", json={"label": label, "frames": synth.sign_clip(seed=5), "learner_id": LEARNER})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_delete_user_sign_removes_files_and_keeps_attempts(client, tmp_root):
    from app.db import ReferenceClip, Session
    from app.registry import registry
    sid = _user_sign(client, "coffee test")
    # A recorder take with a video for the same sign, so there is a video file to clean up too.
    up = client.post("/api/recordings", data={"meta": json.dumps({"sign_id": sid, "signer_id": "me_01"})},
                     files={"landmarks": ("l.json", json.dumps(synth.sign_clip(seed=6)), "application/json"),
                            "video": ("clip.webm", b"fake-webm-bytes", "video/webm")})
    assert up.status_code == 200, up.text
    with Session() as s:
        clips = s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid).all()
        files = [tmp_root / "store" / p for c in clips for p in (c.landmarks_path, c.video_path) if p]
    assert len(clips) == 2 and len(files) == 3 and all(f.is_file() for f in files)
    assert client.post("/api/attempts", json={"learner_id": LEARNER, "sign_id": sid, "score": 70}).status_code == 200

    r = client.delete(f"/api/signs/{sid}")
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": sid, "label": "coffee test", "clips_removed": 2, "files_removed": 3,
                        "attempts_kept": 1}
    assert not any(f.exists() for f in files)
    with Session() as s:
        assert s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid).count() == 0
    # In-memory cache refreshed: the sign is gone from every lookup.
    assert sid not in registry.by_id and "coffee test" not in registry.by_label
    assert client.get(f"/api/signs/{sid}").status_code == 404
    assert all(s["id"] != sid for s in client.get("/api/signs", params={"category": "user"}).json())
    assert client.post("/api/score", json={"sign_id": sid, "frames": synth.sign_clip(seed=5)}).status_code == 404
    # Progress still works and shows the attempt under the deleted sign's label, not due for review.
    p = client.get("/api/progress", params={"learner_id": LEARNER}).json()
    row = next(x for x in p["signs"] if x["sign_id"] == sid)
    assert row["label"] == "coffee test (deleted)" and row["deleted"] is True and row["due"] is False
    assert sid not in p["review_queue"]


def test_delete_builtin_sign_is_forbidden(client):
    letter = next(s for s in client.get("/api/signs", params={"category": "alphabet"}).json() if s["label"] == "A")
    r = client.delete(f"/api/signs/{letter['id']}")
    assert r.status_code == 403 and "built-in" in r.json()["detail"]
    assert client.get(f"/api/signs/{letter['id']}").status_code == 200


def test_delete_missing_sign_is_404(client):
    assert client.delete("/api/signs/999999").status_code == 404


def test_cors_allows_delete(client):
    r = client.options("/api/signs/1", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE"})
    assert r.status_code == 200 and "DELETE" in r.headers["access-control-allow-methods"]
