"""Pick one clean, centred real photo of the handshape per letter and publish it under <store>/media (/media).

    python -m scripts.build_reference_images                     # photos under DATA_DIR (backend/.env)
    python -m scripts.build_reference_images --root <kaggle asl_alphabet_train folder> --force

Only sources whose license allows copying the images are used (PHOTO_SOURCES): the Kaggle "ASL Alphabet" photos
(GPL 2). ChicagoFSVid is not used: no license is stated, and its frames are 128x128 grayscale hand crops.
Candidates are ingested clips whose original photo is still on disk (<root>/<LETTER>/<file>). Each is ranked
within its letter by hand-detection confidence, how centred the hand is, hand size, sharpness, brightness and
contrast; photos with no hand, two hands or a hand touching the edge are skipped. The chosen clip becomes the
sign's canonical clip, so the optional skeleton view shows the same hand. Signs left with neither a video nor a
photo are listed in the report.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.clipstore import load_clip  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import ReferenceClip, ReferenceImage, ReferenceVideo, Session, Sign, init_db  # noqa: E402

PHOTO_SOURCES = {"kaggle_asl_alphabet"}
EXTS = (".jpg", ".jpeg", ".png")
WEIGHTS = {"detection": 1.0, "centred": 1.5, "size": 1.0, "sharpness": 1.0, "brightness": 1.0, "contrast": 0.5}


def photo_path(root: Path, clip: ReferenceClip) -> Path | None:
    """landmarks/<source>/<signer>/<LETTER>/<stem>.npz -> <root>/<LETTER>/<stem>.<ext>"""
    p = Path(clip.landmarks_path)
    for ext in EXTS:
        f = root / p.parent.name / (p.stem + ext)
        if f.is_file():
            return f
    return None


def measures(frame: dict, img: np.ndarray) -> dict | None:
    """Quality measures of one photo (larger is better), or None if it can't be a clean reference."""
    if len(frame.get("hands") or []) != 1:
        return None
    H, W = img.shape[:2]
    pad_x, pad_y = (frame["w"] - W) / 2, (frame["h"] - H) / 2      # ingest pads the photo before detection
    lm = np.asarray(frame["hands"][0]["lm"])[:, :2] * [frame["w"], frame["h"]] - [pad_x, pad_y]
    xy = lm / [W, H]
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    if min(x0, y0, 1 - x1, 1 - y1) < 0.02:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, cv2.convexHull(lm.astype(np.int32)), 255)
    hand = gray[mask > 0]                     # the hand itself, not the (often bright) background
    if hand.size < 50:
        return None
    return {"detection": float(frame["hands"][0].get("score") or 0),
            "centred": -float(np.hypot((x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5)),
            "size": -abs(max(x1 - x0, y1 - y0) - 0.6),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": -abs(float(hand.mean()) - 150),
            "contrast": float(abs(hand.mean() - gray[mask == 0].mean()))}


def best_of(cands: list) -> int:
    """Index of the best candidate: weighted sum of per-measure ranks within the letter."""
    n = len(cands)
    total = np.zeros(n)
    for k, w in WEIGHTS.items():
        v = np.array([m[k] for _, _, m in cands])
        total += w * v.argsort().argsort() / max(n - 1, 1)
    return int(np.argmax(total))


def build(root: Path, force=False, log=print) -> dict:
    init_db()
    st = get_settings()
    report = {"published": [], "kept": 0, "no_photo": [], "sources": sorted(PHOTO_SOURCES)}
    with Session() as s:
        for sign in s.query(Sign).filter(Sign.category == "alphabet").order_by(Sign.label):
            current = s.query(ReferenceImage).filter(ReferenceImage.sign_id == sign.id).one_or_none()
            if current and not force and (st.media_dir / current.path).is_file():
                report["kept"] += 1
                continue
            clips = (s.query(ReferenceClip).filter(ReferenceClip.sign_id == sign.id,
                                                   ReferenceClip.source.in_(PHOTO_SOURCES)).all())
            cands = []
            for c in clips:
                f = photo_path(root, c)
                img = cv2.imread(str(f)) if f else None
                if img is None:
                    continue
                frames, _ = load_clip(st.store_dir / c.landmarks_path)
                m = measures(frames[0], img) if frames else None
                if m:
                    cands.append((c, f, m))
            if not cands:
                report["no_photo"].append(sign.label)
                continue
            clip, src, m = cands[best_of(cands)]
            rel = f"images/{sign.id}/{clip.source}_{src.stem}{src.suffix.lower()}"
            dest = st.media_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            h, w = cv2.imread(str(dest)).shape[:2]
            if current is None:
                current = ReferenceImage(sign_id=sign.id, path=rel)
                s.add(current)
            current.clip_id, current.path, current.width, current.height = clip.id, rel, w, h
            current.license, current.attribution, current.source_url = clip.license, clip.attribution, clip.source_url
            for c in s.query(ReferenceClip).filter(ReferenceClip.sign_id == sign.id):
                c.is_canonical = c.id == clip.id
            report["published"].append({"sign": sign.label, "clip_id": clip.id, "photo": src.name, "path": rel,
                                        "candidates": len(cands), **{k: round(v, 3) for k, v in m.items()}})
            log(f"{sign.label}: {src.name} of {len(cands)} candidates -> /media/{rel} ({clip.license})")
        s.commit()
        with_video = {v.sign_id for v in s.query(ReferenceVideo)}
        with_image = {v.sign_id for v in s.query(ReferenceImage)}
        report["missing_media"] = [{"sign": r.label, "category": r.category, "kind": r.kind}
                                   for r in s.query(Sign).order_by(Sign.category, Sign.label)
                                   if r.id not in with_video and r.id not in with_image]
    out = st.store_dir / "reports" / "reference_images.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    missing = ", ".join(f"{d['sign']} ({d['category']})" for d in report["missing_media"]) or "none"
    log(f"{len(report['published'])} published, {report['kept']} kept. Signs with no video and no photo: {missing}. "
        f"Report: {out}. Restart the API to serve them.")
    return report


def main():
    st = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=st.data_dir,
                    help="folder with one sub-folder of photos per letter (default: DATA_DIR from backend/.env)")
    ap.add_argument("--force", action="store_true", help="re-pick even if a reference photo exists")
    a = ap.parse_args()
    if a.root is None or not a.root.is_dir():
        ap.error("pass --root <folder with A/, B/, ... photo folders> or set DATA_DIR in backend/.env")
    build(a.root, a.force)


if __name__ == "__main__":
    main()
