"""Auto-segmentation (frontend/src/lib/segmenter.js) on scripted motion."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from app.config import REPO_ROOT
from tests import synth

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def _run(frames, kind, tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(frames))
    res = subprocess.run([NODE, "scripts/segmenter_cli.mjs", str(p), kind], capture_output=True, text=True,
                         check=True, cwd=REPO_ROOT / "frontend")
    return json.loads(res.stdout)


def _scripted(rest1=30, move=36, rest2=30, fps=30):
    """Hand at rest, then a fast circle (~1.2 s), then rest again."""
    frames, n = [], rest1 + move + rest2
    for i in range(n):
        if rest1 <= i < rest1 + move:
            u = (i - rest1) / move
            wr = (0.45 + 0.10 * np.cos(2 * np.pi * u), 0.70 + 0.10 * np.sin(2 * np.pi * u))
        else:
            wr = (0.55, 0.70) if i < rest1 else (0.45 + 0.10, 0.70)
        frames.append({"t": i * 1000 / fps, "w": 640, "h": 480, "pose": synth.pose(wr),
                       "hands": [{"handedness": "Left", "score": 0.9, "lm": synth.hand(wr).tolist()}]})
    return frames


def test_dynamic_segment_brackets_the_movement(tmp_path):
    segs = _run(_scripted(), "dynamic", tmp_path)
    assert len(segs) == 1
    start, end, _ = segs[0]
    assert 700 <= start <= 1100          # movement starts at 1000 ms (pre-roll allowed)
    assert 2000 <= end <= 2700           # movement ends at 2200 ms


def test_no_segment_without_motion(tmp_path):
    still = _scripted(rest1=60, move=0, rest2=0)
    assert _run(still, "dynamic", tmp_path) == []


def test_static_hold_emits_once_then_rearms_after_moving(tmp_path):
    still = _scripted(rest1=60, move=0, rest2=0)  # 2 s held still -> one submission, not one per cooldown
    segs = _run(still, "static", tmp_path)
    assert len(segs) == 1 and segs[0][1] - segs[0][0] >= 400
    two = _run(_scripted(rest1=45, move=20, rest2=45), "static", tmp_path)  # hold, move to a new letter, hold
    assert len(two) == 2 and two[1][0] > 45 * 1000 / 30
