import base64
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import func

from . import features as fx
from . import mediapipe_models
from .clipstore import load_clip, quality, save_clip
from .config import get_settings
from .content import letter_hint
from .db import Attempt, Learner, ReferenceClip, Session, Sign, check_connection, init_db
from .recognition import NoHandError, models
from .registry import ensure_alphabet, norm, registry
from .schemas import (AttemptIn, Frames, RecognizeIn, RecognizeOut, ScoreIn, ScoreOut, SignIn, TextIn,
                      frames_to_dicts)
from .text_plan import plan

logger = logging.getLogger("asl.api")
settings = get_settings()
MASTERY_SCORE, MASTERY_OF_LAST = 80.0, (3, 4)


def startup():
    init_db()
    ensure_alphabet()
    registry.load(settings.dev_synthetic)
    models.load()
    settings.media_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(
    title="Handspeak API",
    description="ASL learning coach: recognition, practice scoring and text-to-sign glossing. "
                "A learning aid, not a certified translator; scores are estimates.",
    version="3.0.0",
    lifespan=lifespan,
)


class _TooLarge(Exception):
    pass


class BodyLimit:
    """Reject request bodies over the configured size (multipart uploads get a larger limit)."""

    def __init__(self, app, default: int, upload: int, upload_paths=("/api/recordings",)):
        self.app, self.default, self.upload, self.upload_paths = app, default, upload, upload_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.upload if scope["path"] in self.upload_paths else self.default
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > limit:
            return await JSONResponse({"detail": f"Request body over {limit} bytes."}, 413)(scope, receive, send)
        seen = 0
        started = False

        async def counted():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    raise _TooLarge()
            return msg

        async def tracked_send(msg):
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, counted, tracked_send)
        except _TooLarge:
            if not started:
                await JSONResponse({"detail": f"Request body over {limit} bytes."}, 413)(scope, receive, send)


app.add_middleware(BodyLimit, default=settings.max_body_bytes, upload=settings.max_upload_bytes)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])
app.mount("/media", StaticFiles(directory=settings.media_dir, check_dir=False), name="media")


# ---------------------------------------------------------------- helpers

def _hand_frames(frames) -> int:
    return sum(1 for f in frames if (f.get("hands") if isinstance(f, dict) else f))


def _require_hands(frames, minimum=fx.MIN_HAND_FRAMES):
    if not frames:
        raise HTTPException(422, "No landmark frames provided.")
    if _hand_frames(frames) < minimum:
        raise HTTPException(422, "No hand detected in enough frames. Keep your hands inside the camera frame.")


def _feedback(score: float) -> str:
    if score >= 85:
        return "Excellent. That is very close to the reference."
    if score >= 65:
        return "Good. Check the hints to polish it."
    if score >= 40:
        return "Getting there. Watch the reference again and focus on the first hint."
    return "Not close yet. Follow the hint and try again."


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "sign"


def _touch_learner(s, learner_id: Optional[str]):
    if not learner_id:
        return
    row = s.get(Learner, learner_id)
    if row is None:
        s.add(Learner(id=learner_id, settings={}))
    else:
        row.last_seen = func.now()


def _save_attempt(learner_id, sign_id, score, components, hints, mode, model_version, duration_ms=None):
    try:
        with Session() as s:
            _touch_learner(s, learner_id)
            s.add(Attempt(sign_id=sign_id, score=score, learner_id=learner_id, mode=mode, components=components,
                          hints=hints, model_version=model_version, duration_ms=duration_ms))
            s.commit()
    except Exception as e:  # noqa: BLE001 - scoring must still answer
        logger.warning("Could not save attempt: %s", e)


def _decode_image(data: str):
    import cv2
    if "," in data[:100]:
        data = data.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(data, validate=False), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        img = None
    if img is None:
        raise HTTPException(422, "Could not decode the image.")
    return img


# ---------------------------------------------------------------- status + vocabulary

@app.get("/api/health")
@app.get("/api/status")
def health():
    return {
        "ok": True,
        "database": check_connection(),
        "models": models.status(),
        "feature_spec": fx.SPEC_VERSION,
        "mediapipe": {"version": mediapipe_models.TASKS_VISION_VERSION, "server_models": mediapipe_models.available()},
        "signs_in_cache": len(registry.by_id),
        "dev_synthetic": settings.dev_synthetic,
    }


@app.get("/api/signs")
def list_signs(q: str = "", category: Optional[str] = None, kind: Optional[str] = None,
               has_video: Optional[bool] = None):
    qn = q.strip().lower()
    hits = [e for e in registry.by_id.values()
            if (not qn or qn in e.label.lower() or qn in e.gloss.lower())
            and (not category or e.category == category)
            and (not kind or e.kind == kind)
            and (has_video is None or (e.video is not None) == has_video)]
    hits.sort(key=lambda e: (0 if e.category == "alphabet" else 1, e.label.upper()))
    return [e.public() for e in hits]


def _reference_landmarks(e, max_frames=120):
    with Session() as s:
        clip = None
        if e.canonical_clip_id:
            clip = s.get(ReferenceClip, e.canonical_clip_id)
        if clip is None:
            clip = s.query(ReferenceClip).filter(ReferenceClip.sign_id == e.id).order_by(ReferenceClip.id).first()
    if clip is None:
        return None
    path = settings.store_dir / clip.landmarks_path
    if not path.exists():
        return None
    frames, _ = load_clip(path)
    step = max(1, len(frames) // max_frames + (len(frames) % max_frames > 0))
    return {"frames": frames[::step], "source": clip.source, "signer_id": clip.signer_id,
            "license": clip.license, "attribution": clip.attribution, "clip_id": clip.id}


@app.get("/api/signs/{sign_id}")
def get_sign(sign_id: int):
    e = registry.by_id.get(sign_id)
    if e is None:
        raise HTTPException(404, f"Sign with id {sign_id} not found.")
    d = e.public(settings, with_frames=True)
    d["reference_landmarks"] = _reference_landmarks(e)
    return d


@app.post("/api/signs")
def add_sign(body: SignIn):
    frames = frames_to_dicts(body.frames)
    _require_hands(frames)
    signer = body.learner_id or "anonymous"
    with Session() as s:
        row = Sign(label=body.label, gloss=body.label.upper().replace(" ", "-"), category="user", kind="dynamic",
                   source="user")
        s.add(row)
        s.flush()
        rel = f"landmarks/user/{signer}/{_slug(body.label)}_{row.id}_{int(time.time())}.npz"
        save_clip(settings.store_dir / rel, frames, {"label": body.label, "source": "user", "signer_id": signer})
        s.add(ReferenceClip(sign_id=row.id, source="user", signer_id=signer, landmarks_path=rel,
                            num_frames=len(frames), quality=quality(frames), license="user-contributed (local)",
                            is_canonical=True))
        _touch_learner(s, body.learner_id)
        s.commit()
        sign_id = row.id
    registry.load(settings.dev_synthetic)
    return registry.by_id[sign_id].public()


# ---------------------------------------------------------------- recognition + scoring

@app.post("/api/recognize", response_model=RecognizeOut)
def recognize(body: RecognizeIn):
    frames = frames_to_dicts(body.frames)
    if body.image:
        if not mediapipe_models.available():
            raise HTTPException(503, "Server-side MediaPipe models missing. Run: python -m scripts.download_models")
        from .extract import image_frame
        frames = [image_frame(_decode_image(body.image), pad=40)]
        if not frames[0]["hands"]:
            raise HTTPException(422, "No hand detected in the image.")
    if not frames:
        raise HTTPException(422, "Provide landmark frames or an image.")
    if body.mode == "word":
        return _recognize_word(frames, body)
    model = models.alphabet
    if model is None:
        raise HTTPException(503, "The alphabet model is not trained yet. See README: 'Train the alphabet model'.")
    try:
        p, used = model.predict_frames(frames, body.left_handed)
    except NoHandError as e:
        raise HTTPException(422, str(e)) from e
    order = np.argsort(-p)[:body.top_k]
    matches = []
    for i in order:
        label = model.classes[i]
        e = registry.letter(label)
        matches.append({"label": label, "confidence": round(float(p[i]), 4), "sign_id": e.id if e else None,
                        "reference_video_url": e.video["url"] if e and e.video else None})
    return {"matches": matches, "rejected": bool(p.max() < model.reject_threshold), "method": "alphabet_static",
            "model_version": model.version, "frames_used": used}


def _recognize_word(frames, body):
    wm = models.words
    if wm is None:
        raise HTTPException(503, "The word model is not trained yet. Run: python -m scripts.train_words")
    try:
        seq = wm.sequence(frames, body.left_handed)
    except NoHandError as e:
        raise HTTPException(422, str(e)) from e
    p = wm.proba_seq(seq)
    matches = []
    for i in np.argsort(-p)[:body.top_k]:
        e = registry.by_label.get(norm(wm.classes[i]))
        matches.append({"label": wm.classes[i], "confidence": round(float(p[i]), 4), "sign_id": e.id if e else None,
                        "reference_video_url": e.video["url"] if e and e.video else None})
    return {"matches": matches, "rejected": bool(p.max() < wm.reject_threshold), "method": f"words_{wm.net.kind}",
            "model_version": wm.version, "frames_used": _hand_frames(frames)}


def _legacy_score(frames, legacy_frames):
    """Dev-only: old DTW against synthetic templates (HANDSPEAK_DEV_SYNTHETIC=1)."""
    from .legacy.dtw_v0 import percent, similarity
    from .legacy.features_v0 import to_sequence as to_seq_v0
    old = [[h["lm"] for h in f.get("hands", [])] if isinstance(f, dict) else f for f in frames]
    seq = to_seq_v0(old)
    if seq is None:
        raise HTTPException(422, "No hand detected in the recorded frames.")
    return percent(similarity(seq, np.array(legacy_frames)))


@app.post("/api/score", response_model=ScoreOut)
def score(body: ScoreIn):
    e = registry.by_id.get(body.sign_id)
    if e is None:
        raise HTTPException(404, "Sign not found")
    frames = frames_to_dicts(body.frames)
    _require_hands(frames, minimum=1 if e.kind == "static" else fx.MIN_HAND_FRAMES)
    model = models.alphabet
    if e.category == "alphabet" and e.kind == "static" and model is not None and e.label.upper() in model.classes:
        try:
            p, _ = model.predict_frames(frames, body.left_handed)
        except NoHandError as err:
            raise HTTPException(422, str(err)) from err
        ti = model.classes.index(e.label.upper())
        pred = model.classes[int(np.argmax(p))]
        pct = round(float(p[ti]) * 100, 1)
        hints = letter_hint(e.label, pred if pred != e.label.upper() else None)
        out = {"score": pct, "components": {"handshape": pct}, "hints": hints, "feedback": _feedback(pct),
               "predicted": pred, "sign_label": e.label, "method": "alphabet_static_probability",
               "model_version": model.version}
    elif models.words is not None and e.kind == "dynamic" and e.label in models.words.classes:
        wm = models.words
        try:
            seq = wm.sequence(frames, body.left_handed)
        except NoHandError as err:
            raise HTTPException(422, str(err)) from err
        p = wm.proba_seq(seq)
        r = wm.score(seq, e.label, p)
        pred = wm.classes[int(np.argmax(p))]
        hints = list(r["hints"])
        if pred != e.label and p.max() >= wm.reject_threshold:
            hints.insert(0, f"This looked more like {pred.upper()} than {e.label.upper()}.")
        out = {"score": r["score"], "components": {k: v for k, v in r["components"].items() if v is not None},
               "hints": hints or ["Close match with real signers' productions."], "feedback": _feedback(r["score"]),
               "predicted": pred, "sign_label": e.label, "method": "words_dtw_components", "model_version": wm.version}
    elif settings.dev_synthetic and e.legacy_frames:
        pct = _legacy_score(frames, e.legacy_frames)
        out = {"score": pct, "components": {}, "hints": ["DEV ONLY: compared against a synthetic template."],
               "feedback": _feedback(pct), "predicted": None, "sign_label": e.label,
               "method": "legacy_synthetic_dtw", "model_version": None}
    else:
        raise HTTPException(503, f"No trained scorer for '{e.label}' yet: real reference clips and the dynamic "
                                 "scorer arrive in phases P2–P3.")
    if body.save:
        _save_attempt(body.learner_id, e.id, out["score"], out["components"], out["hints"], body.mode,
                      out["model_version"])
    return out


@app.post("/api/text-to-sign")
def text_to_sign(body: TextIn):
    if not body.text.strip():
        return {"entries": [], "missing": [], "signs": [], "note": None}
    res = plan(body.text, registry)
    # Backward-compatible flat list (one item per sign or letter).
    flat = []
    for ent in res["entries"]:
        ids = [ent["sign_id"]] if not ent["fingerspelled"] else [L["sign_id"] for L in ent["letters"]]
        for sid in ids:
            d = registry.by_id[sid].public(settings, with_frames=True)
            d["is_fingerspelled"] = ent["fingerspelled"]
            flat.append(d)
    res["signs"] = flat
    return res


# ---------------------------------------------------------------- own-data recordings

_frames_adapter = TypeAdapter(Frames)


@app.post("/api/recordings")
async def add_recording(meta: str = Form(...), landmarks: UploadFile = File(...),
                        video: Optional[UploadFile] = File(None)):
    """Save one recorder repetition (landmarks + optional video) as an own-data reference clip."""
    try:
        m = json.loads(meta)
        frames = frames_to_dicts(_frames_adapter.validate_json(await landmarks.read()))
    except (ValueError, ValidationError) as err:
        raise HTTPException(422, f"Invalid recording payload: {err}") from err
    signer = str(m.get("signer_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,64}", signer):
        raise HTTPException(422, "signer_id must be 2-64 letters, digits, '-' or '_'.")
    e = registry.by_id.get(int(m["sign_id"])) if m.get("sign_id") else registry.by_label.get(
        norm(m.get("label") or ""))
    if e is None:
        raise HTTPException(404, "Unknown sign; add it to the vocabulary first.")
    q = quality(frames)
    if q["hand_rate"] < 0.5:
        raise HTTPException(422, f"Hands visible in only {q['hand_rate']:.0%} of frames; recording rejected.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"own/{signer}/{_slug(e.label)}/{stamp}_{int(time.time() * 1000) % 1000:03d}"
    rel_lm = f"landmarks/{base}.npz"
    save_clip(settings.store_dir / rel_lm, frames, {"label": e.label, "source": "own", "signer_id": signer,
                                                    "left_handed": bool(m.get("left_handed")), "rep": m.get("rep")})
    rel_video = None
    if video is not None:
        ext = ".mp4" if "mp4" in (video.content_type or "") else ".webm"
        rel_video = f"recordings/{base}{ext}"
        dst = settings.store_dir / rel_video
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(await video.read())
    ts = [f["t"] for f in frames]
    with Session() as s:
        clip = ReferenceClip(sign_id=e.id, source="own", signer_id=signer, landmarks_path=rel_lm, video_path=rel_video,
                             num_frames=len(frames), start_ms=min(ts), end_ms=max(ts),
                             fps=round(1000 * (len(ts) - 1) / (max(ts) - min(ts)), 2) if max(ts) > min(ts) else None,
                             quality={**q, "left_handed": bool(m.get("left_handed"))},
                             license="own recording (signer consented)", attribution=f"signer {signer}")
        s.add(clip)
        s.commit()
        cid = clip.id
    registry.load(settings.dev_synthetic)
    return {"clip_id": cid, "sign_id": e.id, "quality": q}


@app.get("/api/recordings/summary")
def recording_summary(signer_id: str = Query(..., pattern=r"^[A-Za-z0-9_-]{2,64}$")):
    with Session() as s:
        rows = (s.query(ReferenceClip.sign_id, func.count(ReferenceClip.id))
                .filter(ReferenceClip.source == "own", ReferenceClip.signer_id == signer_id)
                .group_by(ReferenceClip.sign_id).all())
        signers = [r[0] for r in s.query(ReferenceClip.signer_id).filter(ReferenceClip.source == "own").distinct()]
    return {"signer_id": signer_id, "counts": {sid: n for sid, n in rows}, "all_signers": sorted(signers)}


# ---------------------------------------------------------------- attempts + progress

@app.post("/api/attempts")
def add_attempt(body: AttemptIn):
    if body.sign_id not in registry.by_id:
        raise HTTPException(404, "Sign not found")
    _save_attempt(body.learner_id, body.sign_id, body.score, body.components, body.hints, body.mode, None,
                  body.duration_ms)
    return {"ok": True}


@app.get("/api/progress")
def progress(learner_id: str = Query(..., pattern=r"^[A-Za-z0-9_-]{6,64}$")):
    with Session() as s:
        rows = (s.query(Attempt).filter(Attempt.learner_id == learner_id)
                .order_by(Attempt.created_at, Attempt.id).all())
    per = {}
    for a in rows:
        per.setdefault(a.sign_id, []).append(a)
    need, of = MASTERY_OF_LAST
    signs = []
    for sid, atts in per.items():
        e = registry.by_id.get(sid)
        last = [a.score or 0 for a in atts[-of:]]
        signs.append({"sign_id": sid, "label": e.label if e else str(sid), "attempts": len(atts),
                      "best": max(a.score or 0 for a in atts), "last": atts[-1].score,
                      "recent_mean": round(sum(last) / len(last), 1),
                      "mastered": sum(v >= MASTERY_SCORE for v in last) >= need,
                      "last_at": atts[-1].created_at.isoformat() if atts[-1].created_at else None})
    signs.sort(key=lambda d: (d["mastered"], d["recent_mean"]))
    recent = [{"sign_id": a.sign_id, "score": a.score, "mode": a.mode,
               "at": a.created_at.isoformat() if a.created_at else None} for a in rows[-20:]][::-1]
    return {"learner_id": learner_id, "signs": signs, "recent": recent,
            "mastery_rule": f"score ≥ {MASTERY_SCORE:.0f} in {need} of the last {of} attempts"}


# ---------------------------------------------------------------- model files for in-browser inference

@app.get("/api/models/alphabet")
def alphabet_model_file():
    m = models.alphabet
    if m is None or not (m.dir / "browser.json").exists():
        raise HTTPException(404, "Alphabet model not trained yet.")
    return FileResponse(m.dir / "browser.json", media_type="application/json",
                        headers={"Cache-Control": "no-cache"})
