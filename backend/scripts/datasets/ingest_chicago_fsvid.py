"""Ingest ChicagoFSVid (Kim et al. 2017) letter frames as static alphabet clips from 4 signers.

The public release is MATLAB 7.3 files of 128x128 single-channel, background-removed **hand crops** with a letter
label per frame (inside fingerspelled words). MediaPipe barely detects hands in them as-is (14%), so each crop is
tinted to a skin tone on a light background, upscaled and padded (49% detected in a probe). Landmark quality on these
synthetic-looking images is mixed, so clips are tagged domain=hand_crop: they train the model and get their own
cross-signer accuracy, but never count toward "verified" (docs/PLAN.md §7.4). J and Z are skipped: crops follow the
hand, so their motion is lost.

    python -m scripts.datasets.ingest_chicago_fsvid --root D:\\handspeak_data\\downloads\\chicago_fsvid\\extracted\\fsvid

Per letter run (the frames labelled with one letter) the centre frame +-radius is used; a frame is kept if a hand is
detected with score >= --min-score and its handedness label agrees with that signer's majority label.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.datasets.common import add_clip, existing_keys, find_sign, setup  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.content import DYNAMIC_LETTERS  # noqa: E402
from app.db import Session  # noqa: E402
from app.extract import Extractor  # noqa: E402

SOURCE = "chicago_fsvid"
LICENSE = "No license stated by the authors (public research release); research use only, kept local"
ATTRIBUTION = "ChicagoFSVid, Kim et al., Computer Speech and Language 46:209-232 (2017)"
SOURCE_URL = "https://home.ttic.edu/~klivescu/"
SIGNERS = ("andy", "drucie", "rita", "robin")
FPS = 60.0


def letter_runs(labels: np.ndarray):
    """[(start, end, letter_code)] for consecutive frames with one letter label (0-25; 26/27 are silence)."""
    runs, i, n = [], 0, len(labels)
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        if labels[i] < 26:
            runs.append((i, j, int(labels[i])))
        i = j + 1
    return runs


def prep(crop: np.ndarray, scale=2, pad=64, bg=(235, 235, 235)) -> np.ndarray:
    """128x128 hand crop (0-127, background 0) -> BGR image MediaPipe can detect (skin tint, light background)."""
    g = cv2.convertScaleAbs(crop, alpha=2.0)
    mask = crop > 3
    skin = np.stack([g * 0.70, g * 0.80, np.minimum(g * 1.05 + 20, 255)], -1).astype(np.uint8)
    out = np.empty_like(skin)
    out[:] = bg
    out[mask] = skin[mask]
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)


def main():
    import h5py

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="folder with image128_color0_<signer>.mat")
    ap.add_argument("--signers", default=",".join(SIGNERS))
    ap.add_argument("--radius", type=int, default=1, help="frames either side of each letter run's centre")
    ap.add_argument("--max-per-letter", type=int, default=120, help="letter instances per signer (0 = all)")
    ap.add_argument("--min-score", type=float, default=0.8, help="minimum MediaPipe handedness score")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    setup()
    rng = np.random.default_rng(a.seed)
    done = existing_keys(SOURCE)
    report, t0 = {}, time.time()
    with Extractor(mode="image", pose=False) as ex, Session() as s:
        signs = {chr(65 + k): find_sign(s, chr(65 + k), "alphabet") for k in range(26)}
        for signer in a.signers.split(","):
            path = a.root / f"image128_color0_{signer}.mat"
            with h5py.File(path, "r") as f:
                X = f["X"]
                runs = letter_runs(np.array(f["L"]).ravel().astype(int))
                by_letter = collections.defaultdict(list)
                for r in runs:
                    if chr(65 + r[2]) not in DYNAMIC_LETTERS:
                        by_letter[r[2]].append(r)
                picked = []
                for code, rs in sorted(by_letter.items()):
                    if a.max_per_letter and len(rs) > a.max_per_letter:
                        rs = [rs[k] for k in sorted(rng.choice(len(rs), a.max_per_letter, replace=False))]
                    picked += rs
                # Pass 1: detect every selected frame.
                found = []
                for start, end, code in picked:
                    c = (start + end) // 2
                    frames = []
                    for k, idx in enumerate(range(max(start, c - a.radius), min(end, c + a.radius) + 1)):
                        img = prep(np.array(X[idx]).view(np.uint8).T)
                        fr = ex.frame(img, k * 1000 / FPS)
                        fr["hands"] = [h for h in fr["hands"][:1] if h["score"] >= a.min_score]
                        frames.append(fr)
                    found.append((start, code, frames))
            labels = collections.Counter(h["handedness"] for _, _, fr in found for f in fr for h in f["hands"])
            if not labels:
                print(f"{signer}: no hands detected")
                continue
            major = labels.most_common(1)[0][0]
            # MediaPipe labels assume a mirrored image: a majority 'Right' means the hand looks like a left hand in
            # this view, so the signer's clips are mirrored (left_handed) to match everyone else's right hand.
            left_handed = major == "Right"
            per_letter = collections.defaultdict(lambda: [0, 0])
            for start, code, frames in found:
                letter = chr(65 + code)
                per_letter[letter][1] += 1
                keep = [f for f in frames if f["hands"] and f["hands"][0]["handedness"] == major]
                if not keep:
                    continue
                rel = f"landmarks/{SOURCE}/{signer}/{letter}/{start:06d}.npz"
                per_letter[letter][0] += 1
                if rel in done:
                    continue
                add_clip(s, signs[letter], keep, source=SOURCE, signer_id=signer, rel_path=rel, license=LICENSE,
                         attribution=ATTRIBUTION, source_url=SOURCE_URL, fps=FPS,
                         extra_quality={"domain": "hand_crop", "left_handed": left_handed, "frame": start})
            s.commit()
            report[signer] = {"handedness_labels": dict(labels), "left_handed": left_handed,
                              "letters": {k: {"kept": v[0], "instances": v[1]} for k, v in sorted(per_letter.items())}}
            kept, total = sum(v[0] for v in per_letter.values()), sum(v[1] for v in per_letter.values())
            print(f"{signer}: kept {kept}/{total} letter instances ({kept / max(total, 1):.0%}), labels {dict(labels)}"
                  f", mirrored={left_handed} [{time.time() - t0:.0f}s]", flush=True)
    out = get_settings().store_dir / "reports" / "ingest_chicago_fsvid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
