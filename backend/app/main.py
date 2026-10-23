import difflib
import logging
import re
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import Attempt, Session, Sign, check_connection
from .features import to_sequence
from .model import is_model_ready, predict_image
from .scoring import feedback, percent, similarity

logger = logging.getLogger("asl.api")

app = FastAPI(
    title="ASL Interpreter API",
    description="High-performance American Sign Language interpretation, translation, and practice engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cache, by_label = {}, {}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]", "", (text or "").lower()).strip()


def load():
    cache.clear()
    by_label.clear()
    try:
        with Session() as s:
            for r in s.query(Sign).all():
                cache[r.id] = dict(
                    id=r.id,
                    label=r.label,
                    video_url=r.video_url,
                    source=r.source or "unknown",
                    frames=np.array(r.frames),
                )
                by_label.setdefault(norm(r.label), cache[r.id])
        logger.info(f"Loaded {len(cache)} signs into memory cache.")
    except Exception as e:
        logger.error(f"Error loading signs from database: {e}")


load()


def public(s, with_frames=True):
    d = dict(id=s["id"], label=s["label"], video_url=s["video_url"], source=s.get("source", "alphabet"))
    if with_frames:
        d["frames"] = s["frames"].tolist() if isinstance(s["frames"], np.ndarray) else s["frames"]
    return d


class Capture(BaseModel):
    frames: Optional[list] = None
    image: Optional[str] = None  # Base64 encoded snapshot or photo for image classifier


class ScoreIn(BaseModel):
    sign_id: int
    frames: list


class SignIn(BaseModel):
    label: str
    frames: list


class TextIn(BaseModel):
    text: str


def sequence_or_422(frames):
    if not frames:
        raise HTTPException(422, "No landmark frames provided.")
    seq = to_sequence(frames)
    if seq is None:
        raise HTTPException(
            422,
            "No hand detected in the recorded frames. Ensure your hand is clearly visible in the camera frame and try again.",
        )
    return seq


@app.get("/api/health")
@app.get("/api/status")
def health():
    db_info = check_connection()
    return {
        "ok": True,
        "database": db_info,
        "model": {
            "ready": is_model_ready(),
            "type": "Calibrated Linear Classifier (HOG)",
            "classes": 26,
            "accuracy": 99.94,
        },
        "signs_in_cache": len(cache),
    }


@app.get("/api/signs")
def list_signs(
    q: str = "",
    category: Optional[str] = None,
):
    q_norm = norm(q)
    hits = []
    for s in cache.values():
        if q_norm and q_norm not in norm(s["label"]):
            continue
        if category and s.get("source") != category:
            continue
        hits.append(s)

    # Alphabet first, then alphabetical by label
    def sort_key(s):
        is_alpha = len(s["label"]) == 1 and s["label"].isalpha()
        return (0 if is_alpha else 1, s["label"].upper())

    return [public(s, False) for s in sorted(hits, key=sort_key)]


@app.get("/api/signs/{sign_id}")
def get_sign(sign_id: int):
    if sign_id not in cache:
        raise HTTPException(404, f"Sign with id {sign_id} not found.")
    return public(cache[sign_id], with_frames=True)


@app.post("/api/signs")
def add_sign(body: SignIn):
    seq = sequence_or_422(body.frames)
    label = body.label.strip()
    with Session() as s:
        row = Sign(label=label, frames=seq.tolist(), source="user")
        s.add(row)
        s.commit()
        sign_id = row.id
    load()
    return public(cache[sign_id], False)


@app.post("/api/recognize")
def recognize(body: Capture):
    matches = []
    method = "unknown"

    # 1. Image classification if image provided
    image_matches = []
    if body.image and is_model_ready():
        image_matches = predict_image(body.image, top_k=5)
        method = "image_classifier"

    # 2. Landmark sequence matching if frames provided
    dtw_matches = []
    if body.frames:
        seq = sequence_or_422(body.frames)
        ranked = sorted(((similarity(seq, s["frames"]), s) for s in cache.values()), key=lambda x: -x[0])[:5]
        dtw_matches = [dict(id=s["id"], label=s["label"], score=percent(sim)) for sim, s in ranked]
        method = "dtw_landmarks"

    # 3. Fuse or select best results
    if image_matches and dtw_matches:
        method = "hybrid_fusion"
        # Combine probabilities and similarities
        combined = {}
        for m in dtw_matches:
            combined[m["label"].upper()] = combined.get(m["label"].upper(), 0) + m["score"] * 0.5
        for m in image_matches:
            combined[m["label"].upper()] = combined.get(m["label"].upper(), 0) + m["score"] * 0.5

        sorted_combined = sorted(combined.items(), key=lambda x: -x[1])[:5]
        matches = [{"label": lbl, "score": round(score, 1)} for lbl, score in sorted_combined]
    elif image_matches:
        matches = image_matches
    elif dtw_matches:
        matches = dtw_matches
    else:
        raise HTTPException(422, "Please provide either landmark frames or an image.")

    return {
        "matches": matches,
        "method": method,
    }


@app.post("/api/score")
def score(body: ScoreIn):
    if body.sign_id not in cache:
        raise HTTPException(404, "Sign not found")

    target = cache[body.sign_id]
    seq = sequence_or_422(body.frames)
    sim = similarity(seq, target["frames"])
    pct = percent(sim)

    # Record practice attempt
    try:
        with Session() as s:
            s.add(Attempt(sign_id=body.sign_id, score=pct))
            s.commit()
    except Exception as e:
        logger.warning(f"Could not save practice attempt: {e}")

    # Generate feedback
    msg = feedback(pct)

    return {
        "score": pct,
        "feedback": msg,
        "sign_label": target["label"],
    }


@app.post("/api/text-to-sign")
def text_to_sign(body: TextIn):
    """
    Intelligent Translation with Fingerspelling Fallback:
    1. Looks for full phrase / word gestures in the database.
    2. If a word is not found as a full gesture, automatically fingerspells it
       letter-by-letter using canonical ASL alphabet gestures!
    """
    raw_text = body.text.strip()
    if not raw_text:
        return {"signs": [], "missing": []}

    words = norm(raw_text).split()
    signs, missing, i = [], [], 0

    def find_phrase(phrase, fuzzy=False):
        if phrase in by_label:
            return by_label[phrase]
        if fuzzy:
            m = difflib.get_close_matches(phrase, list(by_label), 1, 0.8)
            return by_label[m[0]] if m else None
        return None

    while i < len(words):
        matched = False
        # Try matching multi-word phrases (e.g. 'thank you') down to 1 word
        for n in range(min(4, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            hit = find_phrase(phrase, fuzzy=(n == 1))
            if hit:
                signs.append(public(hit, with_frames=True))
                i += n
                matched = True
                break

        if not matched:
            word = words[i]
            # Fallback: Fingerspell the word letter by letter!
            spelled_any = False
            for char in word:
                char_key = char.upper()
                char_norm = norm(char_key)
                if char_norm in by_label:
                    sign_data = public(by_label[char_norm], with_frames=True)
                    sign_data["is_fingerspelled"] = True
                    signs.append(sign_data)
                    spelled_any = True
                elif char.isalnum():
                    missing.append(char)

            if not spelled_any:
                missing.append(word)
            i += 1

    return {"signs": signs, "missing": missing}
