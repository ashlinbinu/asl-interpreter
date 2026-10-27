"""Import short How2Sign sentence clips (https://how2sign.github.io/, CC BY-NC 4.0) as phrase reference clips.

    python -m scripts.datasets.ingest_how2sign --csv path\to\how2sign_realigned_train.csv ^
        --videos path\to\videos --max-words 3 --limit 300

How2Sign is sentence-level; only very short sentences are useful stand-ins for phrases.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.datasets.common import add_clip, existing_keys, find_sign, setup  # noqa: E402

from app.db import Session, Sign  # noqa: E402
from app.extract import video_frames  # noqa: E402
from app.registry import norm  # noqa: E402

LICENSE = "CC BY-NC 4.0 (How2Sign)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--max-words", type=int, default=3)
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    setup()
    done, added = existing_keys("how2sign"), 0
    with Session() as s, open(a.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            label = norm(row["SENTENCE"])
            video = a.videos / f"{row['SENTENCE_NAME']}.mp4"
            rel = f"landmarks/how2sign/{row['SENTENCE_NAME']}.npz"
            if not label or len(label.split()) > a.max_words or not video.exists() or rel in done:
                continue
            frames, fps = video_frames(video)
            if sum(1 for fr in frames if fr["hands"]) < 4:
                continue
            sign = find_sign(s, label)
            if sign is None:
                sign = Sign(label=label, gloss=label.upper().replace(" ", "-"), category="phrase", kind="dynamic",
                            source="how2sign")
                s.add(sign)
                s.flush()
            add_clip(s, sign, frames, source="how2sign", signer_id=row.get("SENTENCE_NAME", "?").split("-")[0],
                     rel_path=rel, license=LICENSE, attribution="How2Sign (Duarte et al., CVPR 2021)",
                     source_url="https://how2sign.github.io/", fps=fps)
            s.commit()
            added += 1
            print(f"[{added}] {label}")
            if added >= a.limit:
                break
    print("Done. Restart the API to load new clips.")


if __name__ == "__main__":
    main()
