"""Offline MediaPipe extraction plumbing (skipped until scripts.download_models has run)."""
import cv2
import numpy as np
import pytest

from app import mediapipe_models

pytestmark = pytest.mark.skipif(not mediapipe_models.available(), reason="MediaPipe models not downloaded")


def test_image_without_hand_gives_empty_frame():
    from app.extract import image_frame
    f = image_frame(np.full((240, 320, 3), 127, np.uint8))
    assert f["hands"] == [] and f["w"] == 320 and f["h"] == 240


def test_video_pipeline_runs(tmp_path):
    from app.extract import video_frames
    path = tmp_path / "blank.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for i in range(10):
        vw.write(np.full((240, 320, 3), 20 * i, np.uint8))
    vw.release()
    frames, fps = video_frames(path)
    assert len(frames) == 10 and fps == pytest.approx(25, abs=0.5)
    assert all(b["t"] > a["t"] for a, b in zip(frames, frames[1:]))
