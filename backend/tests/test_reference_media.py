"""Real reference media: one photo per letter picked from licensed photos, and media URLs in API responses."""
import cv2
import numpy as np
import pytest

from tests import synth

LETTER = "H"   # one of the synthetic classes of the trained_alphabet fixture


def _photo_clip(s, sign_id, root, store, name, wrist):
    """A 200x200 photo plus its landmarks as the ingest stores them (detected on the photo padded by 40 px)."""
    from app.clipstore import save_clip
    from app.db import ReferenceClip
    (root / LETTER).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / LETTER / f"{name}.jpg"), np.full((200, 200, 3), 150, np.uint8))
    frame = {"t": 0.0, "w": 280, "h": 280, "pose": None,
             "hands": [{"handedness": "Left", "score": 0.95, "lm": synth.hand(wrist, size=0.1).tolist()}]}
    rel = f"landmarks/kaggle_asl_alphabet/test_signer/{LETTER}/{name}.npz"
    save_clip(store / rel, [frame], {"label": LETTER, "source": "kaggle_asl_alphabet", "signer_id": "test_signer"})
    s.add(ReferenceClip(sign_id=sign_id, source="kaggle_asl_alphabet", signer_id="test_signer", landmarks_path=rel,
                        num_frames=1, license="GPL 2 (test)", attribution="test photos"))


@pytest.fixture()
def letter_photo(client, tmp_root):
    from app.db import ReferenceClip, ReferenceImage, Session, Sign
    from app.registry import registry
    from scripts.build_reference_images import build
    root = tmp_root / "photos"
    with Session() as s:
        sid = s.query(Sign).filter(Sign.category == "alphabet", Sign.label == LETTER).one().id
        _photo_clip(s, sid, root, tmp_root / "store", "offcentre", (0.33, 0.5))
        _photo_clip(s, sid, root, tmp_root / "store", "centred", (0.5, 0.62))
        s.commit()
    report = build(root, force=True, log=lambda *_: None)
    registry.load()
    yield sid, report
    with Session() as s:
        s.query(ReferenceImage).filter(ReferenceImage.sign_id == sid).delete()
        s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid, ReferenceClip.source == "kaggle_asl_alphabet").delete()
        s.commit()
    registry.load()


def test_picks_centred_photo_and_serves_it(client, letter_photo):
    sid, report = letter_photo
    chosen = next(p for p in report["published"] if p["sign"] == LETTER)
    assert chosen["photo"] == "centred.jpg" and chosen["candidates"] == 2
    assert {"sign": "J", "category": "alphabet", "kind": "dynamic"} in report["missing_media"]
    d = client.get(f"/api/signs/{sid}").json()
    assert d["reference_image_url"] == f"/media/images/{sid}/kaggle_asl_alphabet_centred.jpg"
    assert d["image_license"] == "GPL 2 (test)" and d["image_attribution"] == "test photos"
    assert d["reference_video_url"] is None
    r = client.get(d["reference_image_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    # The photo's clip becomes canonical, so the optional skeleton shows the same hand.
    assert d["reference_landmarks"]["signer_id"] == "test_signer"


def test_media_urls_in_text_to_sign_and_recognize(client, letter_photo, trained_alphabet):
    sid, _ = letter_photo
    letters = client.post("/api/text-to-sign", json={"text": LETTER.lower()}).json()["entries"][0]["letters"]
    assert letters[0]["image_url"].endswith("centred.jpg") and letters[0]["video_url"] is None
    assert letters[0]["license"] == "GPL 2 (test)"
    frames = [{"t": i * 33.3, "w": 640, "h": 480, "pose": None,
               "hands": [{"handedness": "Left", "score": 0.9, "lm": synth.hand((0.5, 0.6), size=0.12, rot=1.4).tolist()}]}
              for i in range(6)]
    matches = client.post("/api/recognize", json={"frames": frames, "mode": "alphabet"}).json()["matches"]
    m = next(m for m in matches if m["label"] == LETTER)
    assert m["reference_image_url"].endswith("centred.jpg") and "reference_video_url" in m


def test_signs_list_reports_missing_media(client):
    j = next(s for s in client.get("/api/signs", params={"category": "alphabet"}).json() if s["label"] == "J")
    assert j["has_video"] is False and j["has_image"] is False and j["reference_image_url"] is None
