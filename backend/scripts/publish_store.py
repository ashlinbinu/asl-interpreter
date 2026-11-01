"""Copy the vocabulary and the reference media that may be shown publicly into a hosted database and storage
(docs/DEPLOYMENT.md, route B). Run it from backend/ on the machine that holds the full local store:

  python -m scripts.publish_store --to-db "postgresql://USER:PASSWORD@HOST/neondb?sslmode=require" --to-storage s3
  python -m scripts.publish_store --to-db "..." --to-storage s3 --apply

Without --apply it only reports what it would copy and touches nothing. The source is this machine's DATABASE_URL and
HANDSPEAK_STORE_DIR (backend/.env). --to-storage is "s3" (the bucket set by HANDSPEAK_S3_* in the environment or
backend/.env) or a local directory.

Copied: every sign that isn't a custom one (letters and words) and, for those signs, the clips, videos and photos from
--sources with their files. The default sources are the ones whose licenses allow public display (docs/DEPLOYMENT.md,
section 5). Of a sign's clips only the ones the app uses are copied: the canonical clip and the clips its video or
photo came from.
Learners, attempts, custom signs and recordings are never copied. Re-running updates rows in place and uploads only the
files the target doesn't have yet.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, or_  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import ReferenceClip, ReferenceImage, ReferenceVideo, Session, Sign, migrate  # noqa: E402
from app.registry import norm  # noqa: E402
from app.storage import LocalStorage, S3Storage, media_key  # noqa: E402

PUBLIC_SOURCES = ("wikimedia_commons", "kaggle_asl_alphabet")
SIGN_FIELDS = ("label", "gloss", "category", "kind", "hands", "description", "hint_tips", "source")
CLIP_FIELDS = ("source", "signer_id", "split", "landmarks_path", "fps", "num_frames", "start_ms", "end_ms", "quality",
               "license", "attribution", "source_url", "is_canonical")
MEDIA_FIELDS = ("path", "width", "height", "license", "attribution", "source_url")


def plan(src, store, sources=PUBLIC_SOURCES):
    """[(sign, clips, video, image)] to publish. Rows whose file is missing from `store` are left out."""
    items = []
    for sign in src.query(Sign).filter(or_(Sign.category.is_(None), Sign.category != "user")).order_by(Sign.id):
        clips = {c.id: c for c in src.query(ReferenceClip).filter(ReferenceClip.sign_id == sign.id,
                                                                   ReferenceClip.source.in_(sources))}
        video, image = (m if m is not None and m.clip_id in clips and store.exists(media_key(m.path)) else None
                        for m in (src.query(model).filter(model.sign_id == sign.id).one_or_none()
                                  for model in (ReferenceVideo, ReferenceImage)))
        used = {m.clip_id for m in (video, image) if m is not None}
        keep = [c for c in clips.values() if (c.is_canonical or c.id in used) and store.exists(c.landmarks_path)]
        items.append((sign, keep, video, image))
    return items


def files_of(items):
    return [c.landmarks_path for _, clips, _, _ in items for c in clips] + \
        [media_key(m.path) for _, _, video, image in items for m in (video, image) if m is not None]


def publish(src, dst, src_store, dst_store, sources=PUBLIC_SOURCES) -> dict:
    """Copy plan(src) into the dst session and dst_store. Files go first, so no row ever points at a missing file.
    Signs are matched by category and label, so the letters a hosted app has already created are reused."""
    items = plan(src, src_store, sources)
    n = {"signs": 0, "clips": 0, "videos": 0, "images": 0, "files_uploaded": 0}
    for key in files_of(items):
        if not dst_store.exists(key):
            dst_store.write(key, src_store.read(key))
            n["files_uploaded"] += 1
    targets = {(r.category, norm(r.label)): r for r in dst.query(Sign)}
    for sign, clips, video, image in items:
        row = targets.get((sign.category, norm(sign.label)))
        if row is None:
            row = Sign()
            dst.add(row)
        for f in SIGN_FIELDS:
            setattr(row, f, getattr(sign, f))
        dst.flush()
        n["signs"] += 1
        ids = {}
        for c in clips:
            t = dst.query(ReferenceClip).filter(ReferenceClip.sign_id == row.id,
                                                 ReferenceClip.landmarks_path == c.landmarks_path).one_or_none()
            if t is None:
                t = ReferenceClip(sign_id=row.id)
                dst.add(t)
            for f in CLIP_FIELDS:
                setattr(t, f, getattr(c, f))
            t.video_path = None                  # the raw source video stays local
            dst.flush()
            ids[c.id] = t.id
            n["clips"] += 1
        for model, m, name in ((ReferenceVideo, video, "videos"), (ReferenceImage, image, "images")):
            if m is None:
                continue
            t = dst.query(model).filter(model.sign_id == row.id).one_or_none()
            if t is None:
                t = model(sign_id=row.id)
                dst.add(t)
            for f in MEDIA_FIELDS + (("duration_ms",) if model is ReferenceVideo else ()):
                setattr(t, f, getattr(m, f))
            t.clip_id = ids.get(m.clip_id)
            n[name] += 1
    dst.commit()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--to-db", required=True, help="target database URL (the Neon connection string)")
    ap.add_argument("--to-storage", required=True, help='"s3" (bucket from HANDSPEAK_S3_*) or a local directory')
    ap.add_argument("--sources", default=",".join(PUBLIC_SOURCES), help="clip sources whose media may be published")
    ap.add_argument("--apply", action="store_true", help="write to the target (default: report only)")
    a = ap.parse_args()
    sources = tuple(s.strip() for s in a.sources.split(",") if s.strip())
    for s in sorted(set(sources) - set(PUBLIC_SOURCES)):
        print(f"WARNING: '{s}' is not cleared for public display: read docs/DEPLOYMENT.md, section 5, first.")
    src_store = LocalStorage(get_settings().store_dir)
    with Session() as src:
        items = plan(src, src_store, sources)
        keys = files_of(items)
        mb = sum(src_store.path(k).stat().st_size for k in keys) / 2 ** 20
        print(f"{len(items)} signs, {sum(len(c) for _, c, _, _ in items)} clips, "
              f"{sum(v is not None for _, _, v, _ in items)} videos, {sum(i is not None for *_, i in items)} photos: "
              f"{len(keys)} files, {mb:.1f} MB (sources: {', '.join(sources)})")
        if not a.apply:
            print("Dry run: nothing was written. Add --apply to publish.")
            return
        url = "postgresql://" + a.to_db[len("postgres://"):] if a.to_db.startswith("postgres://") else a.to_db
        migrate(url)
        dst_store = S3Storage(**get_settings().s3) if a.to_storage == "s3" else LocalStorage(Path(a.to_storage))
        engine = create_engine(url, pool_pre_ping=True)
        try:
            with sessionmaker(bind=engine)() as dst:
                n = publish(src, dst, src_store, dst_store, sources)
        finally:
            engine.dispose()
    print("Published:", ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in n.items()))


if __name__ == "__main__":
    main()
