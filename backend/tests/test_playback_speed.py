"""Shared playback speed (frontend/src/lib/playback.js): 0.5x is twice as slow as 1x for every kind of player."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

JS = """
import { applyRate, getSpeed, scaled, setSpeed, skeletonFrameIndex, subscribe } from './src/lib/playback.js';
const clip = Array.from({ length: 61 }, (_, i) => ({ t: i * 33.3 }));      // a 2 s landmark clip at 30 fps
const legacy = Array.from({ length: 20 }, () => [0, 0, 0]);                // dev synthetic frames, 80 ms each
const at = (frames, wallMs, speed) => skeletonFrameIndex(frames, wallMs * speed);
const out = {
  clip: { half_1000: at(clip, 1000, 0.5), full_500: at(clip, 500, 1), full_1000: at(clip, 1000, 1) },
  legacy: { half_800: at(legacy, 800, 0.5), full_400: at(legacy, 400, 1) },
  card: [scaled(900, 1), scaled(900, 0.5)], fade: [scaled(150, 1), scaled(150, 0.5)],
};
let calls = 0;
subscribe(() => calls++);
out.initial = getSpeed();
setSpeed(0.5); setSpeed(0.5); setSpeed(3);
out.after = getSpeed(); out.calls = calls;
const video = { playbackRate: 1, defaultPlaybackRate: 1 };
applyRate(video, 0.75);
out.video = video;
process.stdout.write(JSON.stringify(out));
"""


def test_half_speed_is_twice_as_slow_everywhere():
    r = subprocess.run([NODE, "--input-type=module", "-e", JS], capture_output=True, text=True, cwd=FRONTEND)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    # Skeleton: after 1 s at 0.5x it shows the frame that 1x shows after 0.5 s; 1x is twice as far along.
    assert o["clip"]["half_1000"] == o["clip"]["full_500"]
    assert abs(o["clip"]["full_1000"] - 2 * o["clip"]["half_1000"]) <= 1
    assert o["legacy"]["half_800"] == o["legacy"]["full_400"] == 5
    # Sequence cards and the crossfade last twice as long at 0.5x.
    assert o["card"] == [900, 1800] and o["fade"] == [150, 300]
    # One session-wide value: unchanged values and invalid speeds don't notify or change it.
    assert o["initial"] == 1 and o["after"] == 0.5 and o["calls"] == 1
    # Videos: defaultPlaybackRate too, so a new source (which resets playbackRate to it) keeps the speed.
    assert o["video"] == {"playbackRate": 0.75, "defaultPlaybackRate": 0.75}
