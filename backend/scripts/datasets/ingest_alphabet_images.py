"""Ingest a folder-per-letter image dataset (e.g. Kaggle "ASL Alphabet") as static alphabet clips.

Layouts accepted:
  <root>/A/*.jpg ...                         one signer   -> --signer-id required
  <root>/<signer>/<letter>/*.png ...         many signers -> --signers-from-subdirs  (e.g. Surrey/Massey "dataset5")

Each image -> MediaPipe Hand(+Pose) landmarks -> one NPZ + reference_clips row. Images without a detected hand
are discarded, and the per-class detection rate is printed and saved to <store>/reports/.

    python -m scripts.datasets.ingest_alphabet_images --root D:\\data\\asl_alphabet_train --source kaggle_asl_alphabet ^
        --signer-id kaggle_akash --license "see Kaggle dataset page" --per-class 600
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402

from scripts.datasets.common import add_clip, existing_keys, find_sign, setup  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Session  # noqa: E402
from app.extract import Extractor  # noqa: E402

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def letter_dirs(root: Path):
    out = {}
    for d in root.iterdir():
        if d.is_dir() and d.name.upper() in LETTERS:
            out[d.name.upper()] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--source", required=True, help="short dataset id, e.g. kaggle_asl_alphabet")
    ap.add_argument("--license", required=True, help="license string recorded per clip (verify it first)")
    ap.add_argument("--attribution", default=None)
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--signer-id", default=None, help="signer id for a single-signer dataset")
    ap.add_argument("--signers-from-subdirs", action="store_true")
    ap.add_argument("--per-class", type=int, default=500, help="max images per class per signer (random sample)")
    ap.add_argument("--pad", type=int, default=40, help="border (px) added around tight crops before detection")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.signer_id and not a.signers_from_subdirs:
        ap.error("give --signer-id or --signers-from-subdirs")

    setup()
    rng = random.Random(a.seed)
    if a.signers_from_subdirs:
        groups = {d.name: letter_dirs(d) for d in sorted(a.root.iterdir()) if d.is_dir() and letter_dirs(d)}
    else:
        groups = {a.signer_id: letter_dirs(a.root)}
    done = existing_keys(a.source)
    report, t0 = {}, time.time()
    with Extractor(mode="image") as ex, Session() as s:
        for signer, dirs in groups.items():
            for letter in sorted(dirs):
                files = sorted(p for p in dirs[letter].rglob("*") if p.suffix.lower() in EXTS)
                rng.shuffle(files)
                files = files[:a.per_class]
                sign = find_sign(s, letter, "alphabet")
                ok = 0
                for i, f in enumerate(files):
                    rel = f"landmarks/{a.source}/{signer}/{letter}/{f.stem}.npz"
                    if rel in done:
                        ok += 1
                        continue
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    if a.pad:
                        img = cv2.copyMakeBorder(img, a.pad, a.pad, a.pad, a.pad, cv2.BORDER_REPLICATE)
                    frame = ex.frame(img)
                    if not frame["hands"]:
                        continue
                    add_clip(s, sign, [frame], source=a.source, signer_id=signer, rel_path=rel, license=a.license,
                             attribution=a.attribution, source_url=a.source_url)
                    ok += 1
                    if ok % 200 == 0:
                        s.commit()
                s.commit()
                rate = ok / len(files) if files else 0.0
                report.setdefault(signer, {})[letter] = {"images": len(files), "detected": ok,
                                                         "rate": round(rate, 3)}
                print(f"{signer:>16} {letter}: {ok:5d}/{len(files):5d} detected ({rate:.0%})  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    out = get_settings().store_dir / "reports" / f"detection_{a.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Detection report: {out}")


if __name__ == "__main__":
    main()
