import json

import pytest

from tests import synth


def _letter_frames(curl=1.0, rot=0.0, n=6):
    return [{"t": i * 33.3, "w": 640, "h": 480, "pose": None,
             "hands": [{"handedness": "Left", "score": 0.9,
                        "lm": synth.hand((0.5, 0.6), size=0.12, curl=curl, rot=rot).tolist()}]} for i in range(n)]


def _sign_id(client, label):
    return next(s["id"] for s in client.get("/api/signs", params={"q": label}).json() if s["label"] == label)


def test_status_has_no_hardcoded_accuracy(client):
    r = client.get("/api/status").json()
    assert r["ok"] and r["database"]["connected"]
    assert r["feature_spec"] == "1.0"
    assert "accuracy" not in json.dumps(r["models"]["words"])
    assert r["models"]["alphabet"]["ready"] in (True, False)


def test_alphabet_seeded_with_kinds(client):
    letters = client.get("/api/signs", params={"category": "alphabet"}).json()
    assert len(letters) == 26
    kinds = {s["label"]: s["kind"] for s in letters}
    assert kinds["J"] == kinds["Z"] == "dynamic" and kinds["A"] == "static"
    assert all(s["description"] for s in letters)


def test_recognize_requires_model_then_works(client, trained_alphabet):
    metrics, _ = trained_alphabet
    assert metrics["eval_protocol"] == "signer_holdout"
    r = client.post("/api/recognize", json={"frames": _letter_frames(curl=1.0)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["matches"]) == 4  # only four test classes
    top = body["matches"][0]
    assert top["label"] == "A" and top["sign_id"] is not None and 0 <= top["confidence"] <= 1
    assert body["model_version"].startswith("alphabet-")
    status = client.get("/api/status").json()["models"]["alphabet"]
    assert status["ready"] and status["version"] == body["model_version"]
    assert status["accuracy"] == metrics["test"]["accuracy"]


def test_recognize_rejects_empty_and_bad_shapes(client):
    assert client.post("/api/recognize", json={}).status_code == 422
    bad = [{"t": 0, "hands": [{"lm": [[0, 0, 0]] * 20}]}]
    assert client.post("/api/recognize", json={"frames": bad}).status_code == 422


def test_score_letter_gives_hints(client, trained_alphabet):
    sid = _sign_id(client, "B")
    r = client.post("/api/score", json={"sign_id": sid, "frames": _letter_frames(curl=1.0),
                                        "learner_id": "learner-test-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted"] == "A" and body["score"] < 50
    assert any("thumb" in h.lower() or "four fingers" in h.lower() or "looks more like" in h.lower()
               for h in body["hints"])
    good = client.post("/api/score", json={"sign_id": sid, "frames": _letter_frames(curl=0.0)}).json()
    assert good["score"] > body["score"]


def test_add_sign_stores_real_reference_and_is_scored_against_it(client):
    frames = synth.sign_clip(seed=11)
    r = client.post("/api/signs", json={"label": "my test sign", "frames": frames, "learner_id": "learner-test-1"})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    detail = client.get(f"/api/signs/{sid}").json()
    assert detail["category"] == "user" and detail["reference_clips"] == 1
    assert detail["reference_landmarks"]["source"] == "user"
    assert len(detail["reference_landmarks"]["frames"]) == len(frames)
    assert "frames" not in detail  # no synthetic fallback unless HANDSPEAK_DEV_SYNTHETIC=1
    # No trained model covers a custom sign: it is scored against its own recording (DTW), marked unverified.
    r = client.post("/api/score", json={"sign_id": sid, "frames": frames})
    assert r.status_code == 200, r.text
    assert "own_clip_dtw" in r.json()["method"] and "unverified" in r.json()["method"]


def test_text_to_sign_entries_and_fingerspelling(client):
    r = client.post("/api/text-to-sign", json={"text": "my test sign, Bob!"}).json()
    words = [e["word"] for e in r["entries"]]
    assert words == ["my test sign", "bob"]
    bob = r["entries"][1]
    assert bob["fingerspelled"] and [L["letter"] for L in bob["letters"]] == ["B", "O", "B"]
    assert "not ASL grammar" in r["note"]
    assert len(r["signs"]) == 1 + 3


def test_recordings_upload_and_summary(client):
    frames = synth.sign_clip(seed=12)
    sid = _sign_id(client, "L")
    files = {"landmarks": ("lm.json", json.dumps(frames), "application/json"),
             "video": ("v.webm", b"\x1a\x45\xdf\xa3fakewebm", "video/webm")}
    meta = json.dumps({"sign_id": sid, "signer_id": "me_01", "rep": 1})
    r = client.post("/api/recordings", data={"meta": meta}, files=files)
    assert r.status_code == 200, r.text
    assert r.json()["quality"]["hand_rate"] == 1.0
    s = client.get("/api/recordings/summary", params={"signer_id": "me_01"}).json()
    assert s["counts"][str(sid)] == 1 and s["all_signers"] == ["me_01"]
    no_hands = [{**f, "hands": []} for f in frames]
    files["landmarks"] = ("lm.json", json.dumps(no_hands), "application/json")
    assert client.post("/api/recordings", data={"meta": meta}, files=files).status_code == 422


def test_attempts_and_progress_mastery(client):
    sid = _sign_id(client, "C")
    for score in (85, 90, 60, 88):
        assert client.post("/api/attempts", json={"learner_id": "learner-prog", "sign_id": sid,
                                                  "score": score}).status_code == 200
    p = client.get("/api/progress", params={"learner_id": "learner-prog"}).json()
    entry = next(s for s in p["signs"] if s["sign_id"] == sid)
    assert entry["attempts"] == 4 and entry["mastered"] and entry["best"] == 90
    assert entry["box"] == 1 and entry["due"] is False and sid not in p["review_queue"]   # 60 reset the run
    assert client.post("/api/attempts", json={"learner_id": "learner-prog", "sign_id": sid,
                                              "score": 30}).status_code == 200
    p = client.get("/api/progress", params={"learner_id": "learner-prog"}).json()
    entry = next(s for s in p["signs"] if s["sign_id"] == sid)
    assert entry["box"] == 0 and entry["due"] and p["review_queue"][0] == sid


def test_body_size_limit(client):
    big = {"text": "x" * 100, "pad": "y" * (9 * 1024 * 1024)}
    assert client.post("/api/text-to-sign", json=big).status_code == 413


def test_cors_restricted(client):
    ok = client.get("/api/status", headers={"Origin": "http://localhost:5173"})
    bad = client.get("/api/status", headers={"Origin": "http://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers


def test_media_supports_range_requests(client, tmp_root):
    media = tmp_root / "store" / "media" / "videos"
    media.mkdir(parents=True, exist_ok=True)
    (media / "clip.mp4").write_bytes(bytes(range(256)) * 4)
    r = client.get("/media/videos/clip.mp4", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))


def test_browser_model_file(client, trained_alphabet):
    r = client.get("/api/models/alphabet")
    assert r.status_code == 200
    j = r.json()
    assert j["spec_version"] == "1.0" and j["classes"] == ["A", "B", "C", "H"]


@pytest.mark.parametrize("payload", [{"sign_id": 999999, "frames": _letter_frames()}])
def test_score_unknown_sign(client, payload):
    assert client.post("/api/score", json=payload).status_code == 404
