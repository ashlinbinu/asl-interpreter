"""Real reference videos for all 26 letters (J and Z included) from one openly licensed Wikimedia Commons video.

    python -m scripts.datasets.ingest_commons_alphabet            # downloads the 8 MB source if it is missing
    python -m scripts.datasets.ingest_commons_alphabet --force    # cut and publish again

Source: "American Sign Language demo - Travis Dougherty.ogv" (Travis Dougherty fingerspells A-Z; filmed by
Fuzheado), CC BY-SA 3.0, https://commons.wikimedia.org/wiki/File:American_Sign_Language_demo_-_Travis_Dougherty.ogv

Letter times (SPANS) were found by hold detection on the landmarks, read by the alphabet model in A-Z order, and
checked frame by frame; J, P, Q, X and Z (moving or pointing down, so no hold) were placed by eye. Each letter is
cut, cropped to the picture (black side bars removed), MIRRORED to a right-handed view (the signer fingerspells
with his left hand), scaled to 480 p H.264, and published as the letter's reference video. Landmarks are taken
from the published clip, so the skeleton view and scoring match it. CC BY-SA requires attribution and marking the
changes: both are in each clip's attribution. One signer only, so this is a reference to learn from, not test data.
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.clipstore import quality, save_clip  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import ReferenceClip, ReferenceVideo, Session, Sign, init_db  # noqa: E402
from app.extract import video_frames  # noqa: E402

SOURCE = "wikimedia_commons"
SIGNER = "travis_dougherty"
FILE = "American_Sign_Language_demo_-_Travis_Dougherty.ogv"
URL = "https://upload.wikimedia.org/wikipedia/commons/6/64/" + FILE
PAGE = "https://commons.wikimedia.org/wiki/File:" + FILE
SIZE = 8_262_854
LICENSE = "CC BY-SA 3.0"
ATTRIBUTION = ("Travis Dougherty (signer), video by Fuzheado, Wikimedia Commons; cut per letter, cropped and "
               "mirrored to a right-handed view")
CROP = "crop=584:540:149:0"          # the picture inside the 960x540 frame
# Letter -> (start s, end s) in the source video.
SPANS = {
    "A": (17.10, 18.12), "B": (18.12, 19.22), "C": (19.22, 20.09), "D": (20.09, 20.95), "E": (20.95, 22.09),
    "F": (22.09, 23.22), "G": (23.22, 24.05), "H": (24.05, 25.17), "I": (25.17, 25.80), "J": (25.55, 27.10),
    "K": (27.10, 27.89), "L": (27.89, 28.81), "M": (28.81, 29.70), "N": (29.70, 30.58), "O": (30.58, 31.55),
    "P": (31.55, 32.33), "Q": (32.33, 33.10), "R": (33.10, 33.95), "S": (33.95, 34.80), "T": (34.80, 35.60),
    "U": (35.60, 36.28), "V": (36.28, 36.84), "W": (36.84, 37.92), "X": (37.92, 38.78), "Y": (38.78, 39.40),
    "Z": (39.40, 40.35),
}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise SystemExit("Needs imageio-ffmpeg: python -m pip install -r requirements-train.txt") from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch(dest: Path, log=print):
    if dest.is_file() and dest.stat().st_size == SIZE:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {URL} ({SIZE / 1e6:.1f} MB)")
    req = urllib.request.Request(URL, headers={"User-Agent": "Sign-ify/1.0 (local ASL learning app)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest.with_suffix(".part"), "wb") as f:
        shutil.copyfileobj(r, f)
    dest.with_suffix(".part").replace(dest)
    if dest.stat().st_size != SIZE:
        raise SystemExit(f"Unexpected size {dest.stat().st_size} (expected {SIZE}); not using it.")


def cut(src: Path, start: float, end: float, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src),
                    "-vf", f"{CROP},hflip,scale=-2:480", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-an", "-movflags", "+faststart", str(out)], check=True)


def ingest(src: Path, force=False, log=print) -> dict:
    init_db()
    st = get_settings()
    report = {"source": PAGE, "license": LICENSE, "published": [], "kept": []}
    with Session() as s:
        letters = {r.label.upper(): r for r in s.query(Sign).filter(Sign.category == "alphabet")}
        for L, (t0, t1) in SPANS.items():
            sign = letters.get(L)
            if sign is None:
                continue
            existing = (s.query(ReferenceClip).filter(ReferenceClip.sign_id == sign.id, ReferenceClip.source == SOURCE)
                        .one_or_none())
            if existing and not force:
                report["kept"].append(L)
                continue
            raw = f"raw/{SOURCE}/{SIGNER}/{L}.mp4"
            cut(src, t0, t1, st.store_dir / raw)
            frames, fps = video_frames(st.store_dir / raw, max_side=480)
            rel_lm = f"landmarks/{SOURCE}/{SIGNER}/{L}.npz"
            save_clip(st.store_dir / rel_lm, frames, {"label": L, "source": SOURCE, "signer_id": SIGNER,
                                                      "source_span_s": [t0, t1], "mirrored": True})
            clip = existing or ReferenceClip(sign_id=sign.id, source=SOURCE, signer_id=SIGNER, landmarks_path=rel_lm)
            clip.landmarks_path, clip.video_path, clip.num_frames, clip.fps = rel_lm, raw, len(frames), round(fps, 2)
            clip.start_ms, clip.end_ms = frames[0]["t"], frames[-1]["t"]
            clip.quality = {**quality(frames), "domain": "full_frame", "left_handed": False, "mirrored": True}
            clip.license, clip.attribution, clip.source_url = LICENSE, ATTRIBUTION, PAGE
            s.add(clip)
            s.flush()
            for c in s.query(ReferenceClip).filter(ReferenceClip.sign_id == sign.id):
                c.is_canonical = c.id == clip.id          # the skeleton view shows the same hand as the video
            ref = s.query(ReferenceVideo).filter(ReferenceVideo.sign_id == sign.id).one_or_none() or \
                ReferenceVideo(sign_id=sign.id)
            ref.path = f"videos/{sign.id}/{SOURCE}_{clip.id}.mp4"
            (st.media_dir / ref.path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(st.store_dir / raw, st.media_dir / ref.path)
            ref.clip_id, ref.duration_ms = clip.id, round((t1 - t0) * 1000, 1)
            ref.license, ref.attribution, ref.source_url = LICENSE, ATTRIBUTION, PAGE
            s.add(ref)
            report["published"].append({"letter": L, "span_s": [t0, t1], "frames": len(frames),
                                        "hand_rate": clip.quality["hand_rate"], "path": ref.path})
            log(f"{L}: {t0:.2f}-{t1:.2f} s, {len(frames)} frames (hands in {clip.quality['hand_rate']:.0%}) "
                f"-> /media/{ref.path}")
        s.commit()
    out = st.store_dir / "reports" / "commons_alphabet.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"{len(report['published'])} letters published, {len(report['kept'])} kept. Report: {out}. "
        "Restart the API to serve them.")
    return report


def main():
    st = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=st.store_dir / "downloads" / "wikimedia" / FILE)
    ap.add_argument("--force", action="store_true", help="cut and publish again even if already ingested")
    a = ap.parse_args()
    fetch(a.src)
    ingest(a.src, a.force)


if __name__ == "__main__":
    main()
