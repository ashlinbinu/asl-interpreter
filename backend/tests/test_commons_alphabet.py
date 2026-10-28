"""Letter reference videos cut from the Wikimedia Commons alphabet video (CC BY-SA 3.0)."""
import string

import numpy as np
import pytest

from scripts.datasets import ingest_commons_alphabet as ica


def test_letter_spans_cover_the_alphabet_in_order():
    assert list(ica.SPANS) == list(string.ascii_uppercase)
    starts = [t0 for t0, _ in ica.SPANS.values()]
    assert starts == sorted(starts)
    for L, (t0, t1) in ica.SPANS.items():
        assert 0 < t0 < t1 <= 45.4, L                               # inside the 45.5 s source
        assert 0.5 <= t1 - t0 <= 1.6, L                             # about one letter per second
    assert ica.SPANS["J"][1] - ica.SPANS["J"][0] > 1.0              # J and Z include their movement
    assert "CC BY-SA" in ica.LICENSE and "mirrored" in ica.ATTRIBUTION


def test_ingest_publishes_a_letter_video(client, tmp_root, monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    from app import mediapipe_models
    if not mediapipe_models.available():
        pytest.skip("MediaPipe models not downloaded")
    import cv2

    from app.db import ReferenceClip, ReferenceVideo, Session, Sign
    from app.registry import registry
    src = tmp_root / "commons_src.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
    for i in range(60):
        w.write(np.full((540, 960, 3), 40 + i, np.uint8))
    w.release()
    monkeypatch.setattr(ica, "SPANS", {"A": (0.5, 1.5)})
    report = ica.ingest(src, force=True, log=lambda *_: None)
    assert [p["letter"] for p in report["published"]] == ["A"]
    with Session() as s:
        sid = s.query(Sign).filter(Sign.category == "alphabet", Sign.label == "A").one().id
        clip = s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid, ReferenceClip.source == ica.SOURCE).one()
        ref = s.query(ReferenceVideo).filter(ReferenceVideo.sign_id == sid).one()
        canonical = [c.id for c in s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid, ReferenceClip.is_canonical)]
    try:
        assert canonical == [clip.id] and ref.clip_id == clip.id and ref.license == ica.LICENSE
        assert (tmp_root / "store" / "media" / ref.path).is_file() and clip.quality["mirrored"] is True
        registry.load()
        d = client.get(f"/api/signs/{sid}").json()
        assert d["reference_video_url"] == f"/media/{ref.path}" and "mirrored" in d["attribution"]
        video = cv2.VideoCapture(str(tmp_root / "store" / "media" / ref.path))
        assert int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 480        # <= 480 p, cropped and scaled
        video.release()
    finally:
        with Session() as s:
            s.query(ReferenceVideo).filter(ReferenceVideo.sign_id == sid).delete()
            s.query(ReferenceClip).filter(ReferenceClip.sign_id == sid, ReferenceClip.source == ica.SOURCE).delete()
            s.commit()
        registry.load()
