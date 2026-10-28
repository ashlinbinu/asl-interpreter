"""Scoring J and Z, the two letters that move, against a real reference video.

Comparing whole-feature DTW can't tell letters apart (a different letter scores ~94 against a letter's reference),
so J and Z are judged on what defines them:
- handshape: share of frames where only the defining finger is extended (little finger for J, index for Z;
  tip-to-wrist vs knuckle-to-wrist distance), relative to the reference clip;
- movement: the path traced by that fingertip, centred and scaled to its own size, aligned with the reference path
  by DTW and ranked against the reference signer's other 24 letters from the same video ("share of other letters
  whose path is farther away"); a hand that barely moves (under half the reference path) is scaled down.
Score = handshape x movement x 100. One reference signer: an estimate, unverified on other signers; the
leave-one-out check on that signer's letters is in docs/EVALUATION.md.
"""
import string

import numpy as np

from . import features as fx
from .clipstore import load_clip
from .config import get_settings
from .db import ReferenceClip, ReferenceVideo, Session, Sign
from .word_model import dtw_path

TIPS = {"J": 20, "Z": 8}                 # little-finger tip, index tip
ONLY = {"J": 3, "Z": 0}                  # the finger (index, middle, ring, little) that alone is extended
FINGERS = [(5, 8), (9, 12), (13, 16), (17, 20)]
EXTENDED, CURLED = 1.55, 1.35            # tip-to-wrist / knuckle-to-wrist
PATH_POINTS = 32
MIN_REFERENCE_LETTERS = 10


class NoMotionReference(Exception):
    pass


def _hand(frames, left_handed):
    hands, hmask, _, _, t = fx.canonicalize(frames, left_handed)
    ok = hmask[:, 0]
    if ok.sum() < fx.MIN_HAND_FRAMES:
        return None
    return hands[ok, 0], t[ok]


def extension(h):
    """(frames, 4) extension of index, middle, ring and little finger."""
    w = h[:, 0, :2]
    return np.stack([np.linalg.norm(h[:, tip, :2] - w, axis=1)
                     / np.maximum(np.linalg.norm(h[:, mcp, :2] - w, axis=1), 1e-6) for mcp, tip in FINGERS], axis=1)


def describe(frames, letter, left_handed=False):
    """Handshape fraction, path shape and path length (hand sizes) for one clip; None without enough hand."""
    got = _hand(frames, left_handed)
    if got is None:
        return None
    h, t = got
    e = extension(h)
    k = ONLY[letter]
    others = [j for j in range(4) if j != k]
    only = float(((e[:, k] > EXTENDED) & (e[:, others].max(axis=1) < CURLED)).mean())
    size = float(np.median(np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=1))) or 1.0
    pts = h[:, TIPS[letter], :2] / size
    tt = t if t[-1] > t[0] else np.arange(len(pts), dtype=float)
    grid = np.linspace(tt[0], tt[-1], PATH_POINTS)
    path = np.stack([np.interp(grid, tt, pts[:, c]) for c in (0, 1)], axis=1)
    path -= path.mean(axis=0)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    rms = float(np.sqrt((path ** 2).sum(axis=1).mean())) or 1.0
    # Fingertips to highlight: the defining finger if it isn't extended, and any other finger that isn't closed.
    wrong = ({TIPS[letter]} if np.median(e[:, k]) <= EXTENDED else set()) | \
        {FINGERS[j][1] for j in others if np.median(e[:, j]) >= CURLED}
    wrong = sorted(wrong)
    return {"only": only, "shape": path / rms, "length": length, "wrong_joints": wrong}


def _share_farther(d, negatives):
    neg = np.asarray(negatives)
    return float(((neg > d).sum() + 0.5 * (neg == d).sum()) / len(neg))


def _judge(att, ref, neg_dists, letter):
    s_hand = min(1.0, att["only"] / max(ref["only"], 1e-6))
    s_path = _share_farther(dtw_path(att["shape"], ref["shape"])[1], neg_dists)
    gate = min(1.0, att["length"] / (0.5 * ref["length"]))
    return s_hand, s_path * gate, gate


class MotionLetterScorer:
    def __init__(self):
        self._refs = {}

    def invalidate(self):
        self._refs.clear()

    def _clips(self):
        """Clip behind each letter's published reference video."""
        with Session() as s:
            rows = (s.query(Sign.label, ReferenceClip).join(ReferenceClip, ReferenceClip.sign_id == Sign.id)
                    .join(ReferenceVideo, ReferenceVideo.clip_id == ReferenceClip.id)
                    .filter(Sign.category == "alphabet").all())
        return {label.upper(): c for label, c in rows}

    def reference(self, letter):
        """Reference description, the other letters' descriptions and their path distances (cached)."""
        if letter not in self._refs:
            clips = self._clips()
            ref = clips.get(letter)
            if ref is None:
                raise NoMotionReference(letter)
            store = get_settings().store_dir
            same = {L: c for L, c in clips.items() if (c.source, c.signer_id) == (ref.source, ref.signer_id)}
            desc = {}
            for L, c in same.items():
                d = describe(load_clip(store / c.landmarks_path)[0], letter, bool((c.quality or {}).get("left_handed")))
                if d is not None:
                    desc[L] = d
            if letter not in desc or len(desc) <= MIN_REFERENCE_LETTERS:
                raise NoMotionReference(letter)
            r = desc[letter]
            others = {L: d for L, d in desc.items() if L != letter}
            self._refs[letter] = {"ref": r, "others": others, "clip": ref,
                                  "neg": {L: dtw_path(d["shape"], r["shape"])[1] for L, d in others.items()}}
        return self._refs[letter]

    def score(self, frames, letter, left_handed=False) -> dict:
        letter = letter.upper()
        R = self.reference(letter)
        att = describe(frames, letter, left_handed)
        if att is None:
            from .recognition import NoHandError
            raise NoHandError("Not enough frames with a visible hand. Sign inside the frame and try again.")
        s_hand, s_move, gate = _judge(att, R["ref"], list(R["neg"].values()), letter)
        hints = []
        if s_hand < 0.5:
            hints.append("Use the I handshape for J: only the little finger up, the others closed."
                         if letter == "J" else "Use the index finger alone for Z, the other fingers closed.")
        if gate < 0.6:
            hints.append("Move more: J is drawn with the little finger, down then curving toward you."
                         if letter == "J" else "Move more: draw a Z in the air with your index finger.")
        elif s_move < 0.5:
            hints.append(f"Follow the path of {letter} in the reference video"
                         + (" (down, then curve and twist toward you)." if letter == "J"
                            else " (across, diagonally down, across)."))
        highlight = ({"slot": 0, "joints": att["wrong_joints"], "component": "handshape"}
                     if s_hand < 0.5 and att["wrong_joints"] else None)
        return {"score": round(100 * s_hand * s_move, 1),
                "components": {"handshape": round(100 * s_hand, 1), "movement": round(100 * s_move, 1)},
                "hints": hints or [f"Close to the reference {letter}."], "highlight": highlight,
                "reference": {"source": R["clip"].source, "signer_id": R["clip"].signer_id}}

    def leave_one_out(self, letter) -> dict:
        """Each of the reference signer's other letters scored as `letter`, excluding itself from the ranking."""
        R = self.reference(letter)
        out = {}
        for L, d in R["others"].items():
            neg = [v for M, v in R["neg"].items() if M != L]
            s_hand, s_move, _ = _judge(d, R["ref"], neg, letter)
            out[L] = round(100 * s_hand * s_move, 1)
        return dict(sorted(out.items(), key=lambda kv: string.ascii_uppercase.index(kv[0])))


motion_scorer = MotionLetterScorer()
