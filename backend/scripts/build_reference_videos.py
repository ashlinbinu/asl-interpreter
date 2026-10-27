"""Pick one clean reference clip per sign and publish its video under <store>/media (served at /media).

    python -m scripts.build_reference_videos                 # every sign that has clips with a video
    python -m scripts.build_reference_videos --category word --force

Choice: browser-playable codecs (H.264/VP8/VP9/AV1) first, then training-split clips (so validation/test signers
stay unseen by scoring references), then the best capture quality (hands and face visible, hands away from the
frame edge), then the duration closest to the sign's median. The video is copied as-is. If a sign has no playable
clip it is listed in the report (re-encoding with imageio-ffmpeg is planned for P4).
"""
import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import ReferenceClip, ReferenceVideo, Session, Sign, init_db  # noqa: E402

BROWSER_SAFE = {"avc1", "h264", "x264", "vp80", "vp90", "av01"}


def quality_score(c: ReferenceClip) -> float:
    q = c.quality or {}
    return q.get("hand_rate", 0) + 0.5 * q.get("face_rate", 0) - 2.0 * q.get("edge_rate", 0)


def choose(clips: list) -> ReferenceClip | None:
    if not clips:
        return None
    pool = [c for c in clips if c.split == "train"] or clips
    med = statistics.median((c.end_ms or 0) - (c.start_ms or 0) for c in pool)
    return max(pool, key=lambda c: (round(quality_score(c), 2), -abs((c.end_ms or 0) - (c.start_ms or 0) - med),
                                    -c.id))


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {}
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_ms": round(n * 1000 / fps, 1) if fps > 0 else None,
                "codec": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()}
    finally:
        cap.release()


def build(category=None, force=False, log=print) -> dict:
    init_db()
    st = get_settings()
    report = {"published": [], "kept": 0, "not_browser_safe": []}
    with Session() as s:
        signs = s.query(Sign).filter(Sign.category == category) if category else s.query(Sign)
        for sign in signs.order_by(Sign.id):
            clips = [c for c in s.query(ReferenceClip).filter(ReferenceClip.sign_id == sign.id)
                     if c.video_path and (st.store_dir / c.video_path).is_file()]
            if not clips:
                continue
            current = s.query(ReferenceVideo).filter(ReferenceVideo.sign_id == sign.id).one_or_none()
            if current and not force and (st.media_dir / current.path).is_file():
                report["kept"] += 1
                continue
            playable = [c for c in clips if probe(st.store_dir / c.video_path).get("codec") in BROWSER_SAFE]
            best = choose(playable or clips)
            src = st.store_dir / best.video_path
            rel = f"videos/{sign.id}/{best.source}_{best.id}{src.suffix.lower()}"
            dest = st.media_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            info = probe(dest)
            if current is None:
                current = ReferenceVideo(sign_id=sign.id, path=rel)
                s.add(current)
            current.clip_id, current.path = best.id, rel
            current.width, current.height = info.get("width"), info.get("height")
            current.duration_ms = info.get("duration_ms") or ((best.end_ms or 0) - (best.start_ms or 0))
            current.license, current.attribution, current.source_url = best.license, best.attribution, best.source_url
            for c in clips:
                c.is_canonical = c.id == best.id
            entry = {"sign": sign.label, "clip_id": best.id, "split": best.split, "signer": best.signer_id,
                     "path": rel, **info}
            report["published"].append(entry)
            if info.get("codec") and info["codec"] not in BROWSER_SAFE:
                report["not_browser_safe"].append(entry)
            log(f"{sign.label:>14}: clip {best.id} ({best.source}, {best.split}, signer {best.signer_id}) "
                f"-> /media/{rel} [{info.get('codec', '?')}]")
        s.commit()
    out = st.store_dir / "reports" / "reference_videos.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"{len(report['published'])} published, {report['kept']} kept. "
        f"{len(report['not_browser_safe'])} not H.264/VP8/VP9/AV1. Report: {out}. Restart the API to serve them.")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default=None, help="alphabet | word | phrase | user (default: all)")
    ap.add_argument("--force", action="store_true", help="re-pick even if a reference video exists")
    a = ap.parse_args()
    build(a.category, a.force)


if __name__ == "__main__":
    main()
