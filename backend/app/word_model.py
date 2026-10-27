"""Word (dynamic sign) model at runtime: numpy forward pass of the exported 1D-CNN or BiGRU, calibrated top-k, and
per-component scoring against real training-signer references with DTW (docs/PLAN.md P3). Torch is training-only.

Model dir: $HANDSPEAK_MODELS_DIR/words/<version>/{model.json, references.npz, metrics.json}.
"""
import json
from pathlib import Path

import numpy as np

from . import features as fx

COMPONENTS = ("handshape", "orientation", "location", "movement", "two_hand")
BAND = 6                       # Sakoe-Chiba band (frames of the T=32 sequence)
WRIST = slice(94, 96)          # dominant-hand wrist, body coords (FEATURE_SPEC §7)
VEL = slice(122, 124)          # dominant-hand wrist velocity
SECOND_HAND = fx.HAND          # non-dominant present flag


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _conv(x, W, b, pad):
    """x (T, Cin), W (k, Cin, Cout) -> (T, Cout); 'same' padding."""
    k, t = W.shape[0], x.shape[0]
    xp = np.pad(x, ((pad, pad), (0, 0)))
    cols = np.stack([xp[i:i + t] for i in range(k)], axis=1)
    return np.einsum("tki,kio->to", cols, W) + b


def _gru(x, Wih, Whh, bih, bhh, reverse=False):
    """PyTorch GRU equations (gate order r, z, n) over x (T, In) -> (T, H)."""
    H = Whh.shape[1]
    gi = x @ Wih.T + bih
    h, out = np.zeros(H), np.zeros((len(x), H))
    for t in (range(len(x) - 1, -1, -1) if reverse else range(len(x))):
        gh = Whh @ h + bhh
        r = _sigmoid(gi[t, :H] + gh[:H])
        z = _sigmoid(gi[t, H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[t, 2 * H:] + r * gh[2 * H:])
        h = (1 - z) * n + z * h
        out[t] = h
    return out


class WordNet:
    """Numpy forward pass of an exported network (dict from train_words.export)."""

    def __init__(self, d: dict):
        self.kind = d["kind"]
        self.mean, self.std = np.asarray(d["mean"]), np.asarray(d["std"])
        a = lambda v: np.asarray(v, dtype=np.float64)  # noqa: E731
        self.out_W, self.out_b = a(d["out_W"]), a(d["out_b"])
        if self.kind == "cnn":
            self.convs = [(a(c["W"]), a(c["b"]), int(c["pad"])) for c in d["convs"]]
        else:
            self.proj = (a(d["proj_W"]), a(d["proj_b"]))
            self.fw, self.bw = [a(v) for v in d["fw"]], [a(v) for v in d["bw"]]

    def standardize(self, seq):
        return (np.asarray(seq, dtype=np.float64) - self.mean) / self.std

    def logits(self, seq) -> np.ndarray:
        """seq (T, F) raw FEATURE_SPEC features -> (n_classes,) logits."""
        x = self.standardize(seq)
        if self.kind == "cnn":
            for W, b, pad in self.convs:
                x = _relu(_conv(x, W, b, pad))
        else:
            x = _relu(x @ self.proj[0] + self.proj[1])
            x = np.concatenate([_gru(x, *self.fw), _gru(x, *self.bw, reverse=True)], axis=1)
        return np.concatenate([x.mean(0), x.max(0)]) @ self.out_W + self.out_b


def softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def dtw_path(a, b, band=BAND):
    """Banded DTW between (T, D) sequences -> (path [(i, j)], mean step cost)."""
    n, m = len(a), len(b)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    c = np.full((n + 1, m + 1), np.inf)
    c[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            c[i, j] = d[i - 1, j - 1] + min(c[i - 1, j], c[i, j - 1], c[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = ((c[i - 1, j - 1], i - 1, j - 1), (c[i - 1, j], i - 1, j), (c[i, j - 1], i, j - 1))
        _, i, j = min(steps, key=lambda s: s[0])
    path.reverse()
    return path, float(c[n, m] / len(path))


def component_distances(a, b, path) -> dict:
    """Mean per-component distance along an alignment, normalised by the component's size."""
    ia, ib = np.array([p[0] for p in path]), np.array([p[1] for p in path])
    out = {}
    for g in COMPONENTS:
        idx = fx.GROUPS[g]
        diff = a[ia][:, idx] - b[ib][:, idx]
        out[g] = float(np.sqrt((diff ** 2).sum(1)).mean() / np.sqrt(len(idx)))
    return out


class References:
    """Real reference sequences per class (training signers only) + typical same-class distances."""

    def __init__(self, seqs, labels, signers, stats: dict, net: WordNet):
        self.seqs, self.labels, self.signers, self.stats = seqs, list(labels), list(signers), stats
        self.std = net.standardize(seqs)

    def of(self, label):
        return [i for i, l in enumerate(self.labels) if l == label]

    def compare(self, x, label):
        """Best-matching reference of `label` for a standardised sequence x -> (ref index, distances, DTW cost)."""
        best = None
        for i in self.of(label):
            path, cost = dtw_path(x, self.std[i])
            if best is None or cost < best[2]:
                best = (i, component_distances(x, self.std[i], path), cost)
        return best

    def percentiles(self, label, dists) -> dict:
        """Share of real same-sign training attempts that were at least this far from a reference (0-100)."""
        out = {}
        for g, d in dists.items():
            typical = np.asarray(self.stats.get(label, {}).get(g) or [])
            out[g] = round(float((typical >= d).mean()) * 100, 1) if len(typical) else None
        return out


def hints_for(label: str, comp_pct: dict, seq, ref) -> list[str]:
    """Templated, sign-specific hints from the weakest components (direction from the aligned reference)."""
    gloss = label.upper().replace(" ", "-")
    hints = []
    ranked = sorted((v, g) for g, v in comp_pct.items() if v is not None)
    for v, g in ranked[:2]:
        if v >= 50:
            break
        if g == "location":
            d = np.asarray(seq)[:, WRIST].mean(0) - np.asarray(ref)[:, WRIST].mean(0)
            if abs(d[1]) >= abs(d[0]) and abs(d[1]) > 0.12:
                hints.append(f"Your hand is too {'low' if d[1] > 0 else 'high'} for {gloss}: match the reference's "
                             "height (face, chin or chest).")
            elif abs(d[0]) > 0.12:
                hints.append(f"Your hand is too far {'across your body' if d[0] > 0 else 'to your dominant side'} "
                             f"for {gloss}.")
            else:
                hints.append(f"Check where {gloss} is made relative to your face and chest.")
        elif g == "movement":
            sa = np.linalg.norm(np.asarray(seq)[:, VEL], axis=1).mean()
            sr = np.linalg.norm(np.asarray(ref)[:, VEL], axis=1).mean()
            if sr > 1e-6 and sa / sr < 0.6:
                hints.append(f"Make the movement of {gloss} bigger and clearer.")
            elif sr > 1e-6 and sa / sr > 1.7:
                hints.append(f"Make the movement of {gloss} smaller and slower.")
            else:
                hints.append(f"Follow the movement path of {gloss} in the reference video.")
        elif g == "handshape":
            hints.append(f"Your handshape differs most from {gloss}: compare your fingers with the reference.")
        elif g == "orientation":
            hints.append(f"Turn your palm to match the reference for {gloss}.")
        elif g == "two_hand":
            two_ref = np.asarray(ref)[:, SECOND_HAND].mean() > 0.5
            two_you = np.asarray(seq)[:, SECOND_HAND].mean() > 0.5
            hints.append(f"{gloss} uses {'both hands' if two_ref else 'one hand'}." +
                         (" Bring your second hand into view." if two_ref and not two_you else ""))
    return hints


class WordModel:
    def __init__(self, path: Path):
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if d.get("spec_version") != fx.SPEC_VERSION:
            raise ValueError(f"model spec {d.get('spec_version')} != feature spec {fx.SPEC_VERSION}")
        self.net = WordNet(d["net"])
        self.classes = list(d["classes"])
        self.temperature = float(d["temperature"])
        self.reject_threshold = float(d["reject_threshold"])
        self.score_weights = d.get("score_weights")
        self.version = d["version"]
        self.dir = path
        r = np.load(path / "references.npz", allow_pickle=False)
        self.refs = References(r["seqs"], [str(v) for v in r["labels"]], [str(v) for v in r["signers"]],
                               d["component_stats"], self.net)
        mp = path / "metrics.json"
        self.metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}

    def sequence(self, frames, left_handed=False):
        seq = fx.to_sequence(frames, left_handed)
        if seq is None:
            from .recognition import NoHandError
            raise NoHandError("Not enough frames with a visible hand. Sign inside the frame and try again.")
        return seq

    def proba_seq(self, seq) -> np.ndarray:
        return softmax(self.net.logits(seq), self.temperature)

    def score(self, seq, label: str, p: np.ndarray | None = None) -> dict:
        """Score 0-100 for `label`: classifier target probability + component similarity vs. real references."""
        p = self.proba_seq(seq) if p is None else p
        pt = float(p[self.classes.index(label)])
        ri, dists, _ = self.refs.compare(self.net.standardize(seq), label)
        pct = self.refs.percentiles(label, dists)
        w = self.score_weights
        if w:
            x = [pt] + [(pct[g] or 0) / 100 for g in COMPONENTS]
            s = 100 * _sigmoid(float(np.dot(w["coef"], x) + w["intercept"]))
        else:
            vals = [v for v in pct.values() if v is not None]
            s = 50 * pt + 0.5 * (np.mean(vals) if vals else 0)
        return {"score": round(float(s), 1), "target_probability": round(pt, 4), "components": pct,
                "hints": hints_for(label, pct, seq, self.refs.seqs[ri]), "reference_signer": self.refs.signers[ri]}

    def summary(self):
        m = self.metrics
        t = m.get("test", {})
        return {"ready": True, "version": self.version, "type": self.net.kind, "classes": len(self.classes),
                "eval_protocol": m.get("eval_protocol"), "accuracy": t.get("accuracy"),
                "top5_accuracy": t.get("top5_accuracy"), "macro_f1": t.get("macro_f1"),
                "test_signers": len(m.get("signers", {}).get("test", [])), "trained_at": m.get("created_at"),
                "verified": bool(m.get("verified")), "scoring_auc": (m.get("scoring") or {}).get("test_auc")}
