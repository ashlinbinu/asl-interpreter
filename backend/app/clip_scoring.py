"""Scoring for signs that have real reference clips but no trained model: custom signs. (Letters are not scored
this way: against a letter's reference, a different letter by the same signer scores a median of ~94.)

The attempt is aligned with each of the sign's own clips by banded DTW (word_model.dtw_path) and compared per
component (handshape, orientation, location, movement, two hands; FEATURE_SPEC §7). Your attempt is compared with
your own recording, so the yardstick is the same person making a different sign (ASL Citizen signers):
- with the word model and its own_clip_calibration.json (scripts.calibrate_own_clip), features are standardised
  like the word model's, each component becomes "closer than X% of same-person different-sign distances", and a
  monotone logistic fit gives the score; held-out-signer AUC is in that file and docs/EVALUATION.md;
- otherwise raw features are mapped per component through two measured anchors (a same-person repeat's median
  distance -> 75, a same-person different sign's median -> 25) and averaged.
Either way the score is UNVERIFIED for your own signs: it was measured on ASL Citizen repeats, not on custom signs.
"""
import numpy as np

from . import features as fx
from .clipstore import load_clip
from .config import get_settings
from .db import ReferenceClip, Session
from .word_model import COMPONENTS, WEAK, component_distances, dtw_path, highlight_from, hints_for, joints_to_highlight

MAX_REFS = 10
# Median raw-feature distances, ASL Citizen train+val signers: (same signer repeating a word, same signer making a
# different word); 37 and 1118 pairs, measured 2026-10-28.
ANCHORS = {"handshape": (0.308, 0.626), "orientation": (0.257, 0.528), "location": (0.231, 0.452),
           "movement": (0.767, 0.971), "two_hand": (0.074, 0.329)}


class NoUsableReference(Exception):
    pass


def anchored(g, d) -> float:
    """0-100 through a logistic curve that passes 75 at the repeat median and 25 at the different-sign median."""
    pos, neg = ANCHORS[g]
    slope = np.log(3) / ((neg - pos) / 2)
    return round(100 / (1 + float(np.exp(slope * (d - (pos + neg) / 2)))), 1)


def calibrated(cal, dists) -> tuple[dict, float]:
    """Component percentiles vs same-person different-sign distances, and the fitted monotone score."""
    s = [float(1 - np.searchsorted(cal["neg_quantiles"][g], dists[g]) / (len(cal["neg_quantiles"][g]) - 1))
         for g in COMPONENTS]
    x = np.array([np.mean(s)]) if cal.get("combine") == "mean_of_components" else np.array(s)
    p = 1 / (1 + np.exp(-(x @ np.array(cal["coef"]) + cal["intercept"])))
    score = 100 * min(p / cal.get("perfect_probability", 1.0), 1.0)      # a perfect match scores 100
    return {g: round(v * 100, 1) for g, v in zip(COMPONENTS, s)}, round(float(score), 1)


class ClipScorer:
    def __init__(self):
        self._refs = {}       # (sign_id, left_handed) -> [(clip_id, source, signer_id, raw sequence)]
        self._static = {}     # sign_id -> mean static vector of its canonical clip (letters)

    def invalidate(self, sign_id=None):
        """Forget cached references (after a sign's clips change or the sign is deleted)."""
        if sign_id is None:
            self._refs.clear()
            self._static.clear()
            return
        self._refs = {k: v for k, v in self._refs.items() if k[0] != sign_id}
        self._static.pop(sign_id, None)

    def _clips(self, sign_id, limit=MAX_REFS):
        with Session() as s:
            return (s.query(ReferenceClip).filter(ReferenceClip.sign_id == sign_id)
                    .order_by(ReferenceClip.is_canonical.desc(), ReferenceClip.id).limit(limit).all())

    def references(self, sign_id, left_handed=False):
        key = (sign_id, bool(left_handed))
        if key not in self._refs:
            store, refs = get_settings().store_dir, []
            for c in self._clips(sign_id):
                path = store / c.landmarks_path
                if not path.is_file():
                    continue
                frames, _ = load_clip(path)
                # Custom signs were recorded by this learner, so they share the learner's handedness setting.
                lh = left_handed if c.source == "user" else bool((c.quality or {}).get("left_handed"))
                seq = fx.to_sequence(frames, lh)
                if seq is not None:
                    refs.append((c.id, c.source, c.signer_id, seq))
            self._refs[key] = refs
        return self._refs[key]

    def score(self, frames, sign_id, label, left_handed=False, wm=None) -> dict:
        refs = self.references(sign_id, left_handed)
        if not refs:
            raise NoUsableReference(label)
        x = fx.to_sequence(frames, left_handed)
        if x is None:
            from .recognition import NoHandError
            raise NoHandError("Not enough frames with a visible hand. Sign inside the frame and try again.")
        cal = getattr(wm, "own_clip", None) if wm is not None else None
        std = wm.net.standardize if cal else (lambda q: q)
        xs = std(x)
        best = None
        for cid, source, signer, r in refs:
            rs = std(r)
            path, cost = dtw_path(xs, rs)
            if best is None or cost < best[0]:
                best = (cost, path, r, rs, cid, source, signer)
        _, path, r, rs, cid, source, signer = best
        dists = component_distances(xs, rs, path)
        if cal:
            pct, total = calibrated(cal, dists)
            method = "own_clip_dtw"
        else:
            pct = {g: anchored(g, d) for g, d in dists.items()}
            total = round(float(np.mean(list(pct.values()))), 1)
            method = "own_clip_dtw_anchored"
        weak_shape = pct["handshape"] < WEAK
        return {"score": total, "components": pct, "hints": hints_for(label, pct, x, r),
                "highlight": highlight_from(x, r, path) if weak_shape else None, "method": method,
                "references": len(refs), "reference": {"clip_id": cid, "source": source, "signer_id": signer}}

    def letter_highlight(self, frames, sign_id, left_handed=False):
        """Joints of the attempt's hand that differ most from the letter's reference photo (canonical clip)."""
        if sign_id not in self._static:
            vec, store = None, get_settings().store_dir
            clips = self._clips(sign_id, limit=1)
            if clips and (store / clips[0].landmarks_path).is_file():
                v, _ = fx.static_vectors(load_clip(store / clips[0].landmarks_path)[0],
                                         bool((clips[0].quality or {}).get("left_handed")))
                vec = v.mean(0) if len(v) else None
            self._static[sign_id] = vec
        ref = self._static[sign_id]
        att, _ = fx.static_vectors(frames, left_handed)
        if ref is None or not len(att):
            return None
        err = np.linalg.norm(att.mean(0)[:63].reshape(21, 3) - ref[:63].reshape(21, 3), axis=1)
        return joints_to_highlight(err, 0)


clip_scorer = ClipScorer()
