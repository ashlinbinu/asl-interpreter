"""In-memory view of the vocabulary (signs + reference availability), loaded once at startup."""
import re
import string
from dataclasses import dataclass, field

from sqlalchemy import func

from .content import DYNAMIC_LETTERS, LETTERS
from .db import ReferenceClip, ReferenceImage, ReferenceVideo, Session, Sign


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]", "", (text or "").lower().replace("-", " ")).strip()


@dataclass
class SignEntry:
    id: int
    label: str
    gloss: str
    category: str
    kind: str
    description: str | None
    video: dict | None = None            # {url, license, attribution, source_url}
    image: dict | None = None            # same shape: a real photo of the handshape
    clip_count: int = 0
    canonical_clip_id: int | None = None
    legacy_frames: list | None = field(default=None, repr=False)
    frames_source: str | None = None

    def public(self, settings=None, with_frames=False):
        d = {"id": self.id, "label": self.label, "gloss": self.gloss, "category": self.category,
             "kind": self.kind, "description": self.description,
             "has_video": self.video is not None, "reference_video_url": self.video["url"] if self.video else None,
             "license": self.video.get("license") if self.video else None,
             "attribution": self.video.get("attribution") if self.video else None,
             "has_image": self.image is not None, "reference_image_url": self.image["url"] if self.image else None,
             "image_license": self.image.get("license") if self.image else None,
             "image_attribution": self.image.get("attribution") if self.image else None,
             "reference_clips": self.clip_count}
        if with_frames and settings is not None and settings.dev_synthetic and self.legacy_frames:
            d["frames"] = self.legacy_frames
            d["frames_source"] = self.frames_source
        return d


class Registry:
    def __init__(self):
        self.by_id: dict[int, SignEntry] = {}
        self.by_label: dict[str, SignEntry] = {}

    def load(self, dev_synthetic=False):
        by_id, by_label = {}, {}
        with Session() as s:
            clip_counts = dict(s.query(ReferenceClip.sign_id, func.count(ReferenceClip.id))
                               .group_by(ReferenceClip.sign_id).all())
            canon = dict(s.query(ReferenceClip.sign_id, ReferenceClip.id).filter(ReferenceClip.is_canonical.is_(True)))
            videos = {v.sign_id: v for v in s.query(ReferenceVideo)}
            images = {v.sign_id: v for v in s.query(ReferenceImage)}
            media = lambda v: {"url": f"/media/{v.path}", "license": v.license,  # noqa: E731
                               "attribution": v.attribution, "source_url": v.source_url} if v else None
            for r in s.query(Sign).order_by(Sign.id):
                e = SignEntry(
                    id=r.id, label=r.label, gloss=r.gloss or r.label.upper(), category=r.category or "word",
                    kind=r.kind or "dynamic", description=r.description,
                    video=media(videos.get(r.id)), image=media(images.get(r.id)),
                    clip_count=clip_counts.get(r.id, 0), canonical_clip_id=canon.get(r.id),
                    legacy_frames=r.frames if dev_synthetic else None, frames_source=r.frames_source,
                )
                by_id[e.id] = e
                key = norm(e.label)
                # Prefer real vocabulary entries over user-added duplicates for text lookup.
                if key not in by_label or by_label[key].category == "user":
                    by_label[key] = e
        self.by_id, self.by_label = by_id, by_label

    def letter(self, ch: str) -> SignEntry | None:
        e = self.by_label.get(ch.lower())
        return e if e and e.category == "alphabet" else None


def ensure_alphabet():
    """Make sure A-Z exist as vocabulary entries (with descriptions) regardless of seeding."""
    with Session() as s:
        existing = {r.label.upper(): r for r in s.query(Sign).filter(Sign.category == "alphabet")}
        for ch in string.ascii_uppercase:
            row = existing.get(ch)
            if row is None:
                row = Sign(label=ch, category="alphabet", source="alphabet")
                s.add(row)
            row.gloss = ch
            row.kind = "dynamic" if ch in DYNAMIC_LETTERS else "static"
            row.hands = "one"
            row.description = LETTERS[ch]
        s.commit()


registry = Registry()
