"""Build the sign library from How2Sign (https://how2sign.github.io/).

1. Download a translation CSV (e.g. how2sign_realigned_train.csv) and the matching
   RGB front-view sentence clips (<SENTENCE_NAME>.mp4) from the How2Sign site.
2. From /backend run:
   python -m scripts.ingest_how2sign --csv path/to/how2sign_realigned_train.csv --videos path/to/videos --max-words 3 --limit 300
How2Sign is sentence level, so short phrases are the best stand-ins for words.
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

from app.db import Session, Sign
from app.features import to_sequence
from app.main import norm


def extract(path):
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frames.append([[[p.x, p.y, p.z] for p in h.landmark] for h in (res.multi_hand_landmarks or [])])
    cap.release(); hands.close()
    return frames


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--videos", required=True)
ap.add_argument("--max-words", type=int, default=3)
ap.add_argument("--limit", type=int, default=300)
a = ap.parse_args()

seen, added = set(), 0
with Session() as db, open(a.csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        label = norm(row["SENTENCE"])
        video = Path(a.videos) / f"{row['SENTENCE_NAME']}.mp4"
        if not label or label in seen or len(label.split()) > a.max_words or not video.exists():
            continue
        seq = to_sequence(extract(video))
        if seq is None:
            continue
        db.add(Sign(label=label, frames=seq.tolist(), source="how2sign")); db.commit()
        seen.add(label); added += 1
        print(f"[{added}] {label}")
        if added >= a.limit:
            break
print("Done. Restart the API to load new signs.")
