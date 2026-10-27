"""Record your own reference/training clips from the webcam (OpenCV window).

    python -m scripts.record_cli --signer-id me_01 --signs A B C --reps 5
    python -m scripts.record_cli --signer-id me_01 --alphabet --reps 5 --seconds 1.5
    python -m scripts.record_cli --signer-id me_01 --signs hello "thank you" --seconds 3 --left-handed

Per repetition: 3-2-1 countdown, record, quality check (hands visible, face visible for words), save
landmarks NPZ + MP4 under $HANDSPEAK_STORE_DIR and a reference_clips row (source "own").
Keys: SPACE = accept/next, R = redo last, S = skip sign, Q = quit.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from scripts.datasets.common import add_clip, find_sign, setup  # noqa: E402

from app.clipstore import quality  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import Session  # noqa: E402
from app.extract import Extractor  # noqa: E402

HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
             (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw(img, frame):
    h, w = img.shape[:2]
    for hd in frame["hands"]:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hd["lm"]]
        for a, b in HAND_CONN:
            cv2.line(img, pts[a], pts[b], (0, 230, 255), 2)
    if frame["pose"]:
        for j in (0, 2, 5, 9, 10, 11, 12, 13, 14):
            x, y, _, v = frame["pose"][j]
            if v > 0.5:
                cv2.circle(img, (int(x * w), int(y * h)), 3, (255, 255, 255), -1)


def banner(img, text, sub=""):
    disp = cv2.flip(img, 1)  # mirror for display only; data stays un-mirrored
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(disp, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2)
    if sub:
        cv2.putText(disp, sub, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.imshow("Handspeak recorder", disp)
    return cv2.waitKey(1) & 0xFF


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer-id", required=True)
    ap.add_argument("--signs", nargs="*", default=[])
    ap.add_argument("--alphabet", action="store_true", help="record A-Z (J and Z as 3 s motion takes)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=None, help="default 1.5 s for letters, 3 s for words")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--left-handed", action="store_true")
    a = ap.parse_args()
    labels = list(a.signs) + ([chr(c) for c in range(ord("A"), ord("Z") + 1)] if a.alphabet else [])
    if not labels:
        ap.error("give --signs or --alphabet")
    setup()
    store = get_settings().store_dir
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")
    t_origin = time.monotonic()
    with Extractor(mode="video") as ex, Session() as s:
        for label in labels:
            sign = find_sign(s, label)
            if sign is None:
                print(f"skip '{label}': not in the vocabulary")
                continue
            is_letter = sign.category == "alphabet"
            secs = a.seconds or (1.5 if is_letter and sign.kind == "static" else 3.0)
            rep = 0
            while rep < a.reps:
                t_end = time.monotonic() + 3
                while time.monotonic() < t_end:  # countdown with live preview
                    ok, img = cap.read()
                    if not ok:
                        continue
                    fr = ex.frame(img, (time.monotonic() - t_origin) * 1000)
                    draw(img, fr)
                    if banner(img, f"{sign.label}  rep {rep + 1}/{a.reps}", f"starting in {t_end - time.monotonic():.1f}s"
                              ) == ord("q"):
                        return
                frames, video, t0 = [], [], time.monotonic()
                while time.monotonic() - t0 < secs:
                    ok, img = cap.read()
                    if not ok:
                        continue
                    fr = ex.frame(img, (time.monotonic() - t_origin) * 1000)
                    frames.append(fr)
                    video.append(img.copy())
                    draw(img, fr)
                    banner(img, f"RECORDING {sign.label}", f"{secs - (time.monotonic() - t0):.1f}s")
                q = quality(frames)
                ok_q = q["hand_rate"] >= (0.8 if is_letter else 0.6) and (is_letter or q["face_rate"] >= 0.5)
                msg = (f"hands {q['hand_rate']:.0%}  face {q['face_rate']:.0%}  edge {q['edge_rate']:.0%}  -> "
                       + ("SPACE accept, R redo" if ok_q else "LOW QUALITY: R redo, SPACE keep anyway"))
                key = 0
                while key not in (ord(" "), ord("r"), ord("s"), ord("q")):
                    key = banner(video[-1] if video else img, f"{sign.label} done", msg)
                if key == ord("q"):
                    return
                if key == ord("s"):
                    break
                if key == ord("r"):
                    continue
                stamp = time.strftime("%Y%m%d-%H%M%S")
                base = f"own/{a.signer_id}/{sign.label.lower().replace(' ', '-')}/{stamp}_{rep}"
                rel_video = f"recordings/{base}.mp4"
                vpath = store / rel_video
                vpath.parent.mkdir(parents=True, exist_ok=True)
                fps = len(frames) / secs
                vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (video[0].shape[1], video[0].shape[0]))
                for v in video:
                    vw.write(v)
                vw.release()
                add_clip(s, sign, frames, source="own", signer_id=a.signer_id, rel_path=f"landmarks/{base}.npz",
                         license="own recording (signer consented)", attribution=f"signer {a.signer_id}",
                         video_path=rel_video, fps=round(fps, 2), extra_quality={"left_handed": a.left_handed})
                s.commit()
                rep += 1
                print(f"saved {sign.label} rep {rep}: {q}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
