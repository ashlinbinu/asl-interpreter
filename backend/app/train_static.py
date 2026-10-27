"""Static (alphabet) classifier training: signer-grouped splits, RF vs MLP, calibration, honest metrics.

Inputs are clips: dicts {frames, label, signer_id, clip_id, left_handed}. Every frame with a hand becomes a sample;
splits are always made by signer (or, with a single signer, by contiguous clip blocks, clearly flagged).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as fx
from .content import CONFUSABLE_SETS, DYNAMIC_LETTERS
from .recognition import calibrate


# ---------------------------------------------------------------- augmentation (raw landmark level)

def augment_frame(f: dict, rng: np.random.Generator, rot_deg=12.0, noise=0.004) -> dict:
    """Small in-plane rotation about the wrist + landmark noise. Training only."""
    a = (f.get("w") or 640) / (f.get("h") or 480)
    th = math.radians(rng.uniform(-rot_deg, rot_deg))
    c, s = math.cos(th), math.sin(th)
    hands = []
    for hd in f.get("hands") or []:
        lm = np.asarray(hd["lm"], float)
        x, y = lm[:, 0] * a, lm[:, 1]
        x0, y0 = x[0], y[0]
        xr = x0 + (x - x0) * c - (y - y0) * s
        yr = y0 + (x - x0) * s + (y - y0) * c
        out = np.stack([xr / a, yr, lm[:, 2]], axis=1) + rng.normal(0, noise, (21, 3))
        hands.append({**hd, "lm": out.tolist()})
    return {**f, "hands": hands}


def clip_samples(clip, stride=1, rng=None, n_aug=0):
    """-> (M, 93) static vectors for one clip (+ augmented copies)."""
    frames = clip["frames"][::stride]
    vecs, _ = fx.static_vectors(frames, clip.get("left_handed", False))
    out = [vecs]
    for _ in range(n_aug):
        aug = [augment_frame(f, rng) for f in frames]
        v, _ = fx.static_vectors(aug, clip.get("left_handed", False))
        out.append(v)
    return np.concatenate(out) if out else np.zeros((0, fx.STATIC_DIM))


# ---------------------------------------------------------------- splits

def make_splits(clips, test_signers=None, val_signers=None):
    """Assign train/val/test per clip by signer. Returns (assignments, protocol, warning)."""
    signers = sorted({c["signer_id"] for c in clips})
    if len(signers) >= 3:
        test = list(test_signers or [signers[-1]])
        rest = [s for s in signers if s not in test]
        val = list(val_signers or [rest[-1]])
        train = [s for s in rest if s not in val]
        assign = {c["clip_id"]: ("test" if c["signer_id"] in test else "val" if c["signer_id"] in val else "train")
                  for c in clips}
        return assign, {"protocol": "signer_holdout", "train": train, "val": val, "test": test}, None
    if len(signers) == 2:
        test = list(test_signers or [signers[-1]])
        train_s = [s for s in signers if s not in test]
        assign = {}
        for label in {c["label"] for c in clips}:
            own = sorted([c for c in clips if c["label"] == label and c["signer_id"] in train_s],
                         key=lambda c: str(c["clip_id"]))
            cut = int(len(own) * 0.8)
            for i, c in enumerate(own):
                assign[c["clip_id"]] = "train" if i < cut or len(own) < 3 else "val"
        for c in clips:
            if c["signer_id"] in test:
                assign[c["clip_id"]] = "test"
        return assign, {"protocol": "signer_holdout_test_only", "train": train_s, "val": train_s, "test": test}, \
            "Only 2 signers: validation/calibration is within the training signer."
    # One signer: contiguous clip blocks per class (reduces, but does not remove, near-duplicate leakage).
    assign = {}
    for label in {c["label"] for c in clips}:
        own = sorted([c for c in clips if c["label"] == label], key=lambda c: str(c["clip_id"]))
        n = len(own)
        for i, c in enumerate(own):
            assign[c["clip_id"]] = "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test"
    return assign, {"protocol": "single_signer_block_split", "train": signers, "val": signers, "test": signers}, \
        "SINGLE SIGNER: test accuracy is NOT a signer-independent estimate and will overstate live accuracy."


# ---------------------------------------------------------------- models

def candidates(seed=0):
    return {
        "rf": RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=2,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                                             learning_rate_init=2e-3, max_iter=300,
                                                             early_stopping=True, n_iter_no_change=15,
                                                             random_state=seed)),
    }


def _balanced(X, y, rng):
    """Oversample minority classes to the majority count (MLP has no class weights)."""
    labels, counts = np.unique(y, return_counts=True)
    m = counts.max()
    idx = np.concatenate([rng.choice(np.flatnonzero(y == L), m, replace=len(np.flatnonzero(y == L)) < m)
                          for L in labels])
    return X[idx], y[idx]


def _proba(est, X, classes):
    p = est.predict_proba(X)
    order = [list(est.classes_).index(c) for c in classes]
    return p[:, order]


def fit_temperature(p, y_idx):
    best_t, best = 1.0, np.inf
    for t in np.geomspace(0.25, 8.0, 64):
        q = calibrate(p, t)
        nll = -np.mean(np.log(np.clip(q[np.arange(len(y_idx)), y_idx], 1e-12, 1)))
        if nll < best:
            best_t, best = float(t), nll
    return best_t


def ece(p, y_idx, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(e)


def reject_threshold(p, y_idx, target_precision=0.95):
    """Smallest max-prob threshold whose accepted predictions reach the target precision on validation."""
    conf, ok = p.max(1), p.argmax(1) == y_idx
    for t in np.linspace(0.0, 0.95, 96):
        m = conf >= t
        if m.sum() >= max(10, 0.2 * len(conf)) and ok[m].mean() >= target_precision:
            return float(t), float(m.mean())
    return 0.95, float((conf >= 0.95).mean())


def report(y_true, y_pred, classes):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pairs = []
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": a, "pred": b, "count": int(cm[i, j]),
                              "rate": round(cm[i, j] / max(cm[i].sum(), 1), 3)})
    pairs.sort(key=lambda d: -d["rate"])
    watched = [{"set": "".join(sset), "accuracy": round(float(np.mean([y_pred[k] == y_true[k]
                for k in range(len(y_true)) if y_true[k] in sset])), 3)}
               for sset in CONFUSABLE_SETS if any(t in sset for t in y_true)]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)), 4),
        "n": int(len(y_true)),
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "f1": round(float(f[i]), 3), "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion": {"labels": classes, "matrix": cm.tolist()},
        "top_confusions": pairs[:15],
        "confusable_sets": watched,
    }


def export_browser(est, kind, classes, temperature, threshold, version):
    base = {"kind": kind, "classes": classes, "temperature": temperature, "reject_threshold": threshold,
            "spec_version": fx.SPEC_VERSION, "version": version, "input_dim": fx.STATIC_DIM}
    if kind == "mlp":
        scaler, mlp = est.steps[0][1], est.steps[-1][1]
        order = [list(mlp.classes_).index(c) for c in classes]
        layers = [{"W": np.round(W, 6).tolist(), "b": np.round(b, 6).tolist()}
                  for W, b in zip(mlp.coefs_, mlp.intercepts_)]
        layers[-1]["W"] = np.round(mlp.coefs_[-1][:, order], 6).tolist()
        layers[-1]["b"] = np.round(mlp.intercepts_[-1][order], 6).tolist()
        return {**base, "scaler": {"mean": scaler.mean_.round(6).tolist(), "scale": scaler.scale_.round(6).tolist()},
                "layers": layers, "activation": mlp.activation}
    order = [list(est.classes_).index(c) for c in classes]
    trees = []
    for t in est.estimators_:
        tr = t.tree_
        leaves = {}
        for node in np.flatnonzero(tr.children_left == -1):
            v = tr.value[node][0][order]
            v = v / v.sum()
            leaves[int(node)] = {int(k): round(float(v[k]), 3) for k in np.flatnonzero(v > 0.005)}
        trees.append({"f": tr.feature.tolist(), "t": tr.threshold.tolist(),
                      "l": tr.children_left.tolist(), "r": tr.children_right.tolist(), "leaf": leaves})
    return {**base, "trees": trees}


# ---------------------------------------------------------------- main entry

def train(clips, out_root: Path, sources=None, test_signers=None, val_signers=None, seed=0, n_aug=2, stride=1,
          log=print):
    rng = np.random.default_rng(seed)
    clips = [c for c in clips if c["label"].upper() not in DYNAMIC_LETTERS]
    for c in clips:
        c["label"] = c["label"].upper()
    assign, signers, warning = make_splits(clips, test_signers, val_signers)
    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for c in clips:
        part = assign[c["clip_id"]]
        X = clip_samples(c, stride, rng, n_aug if part == "train" else 0)
        data[part][0].append(X)
        data[part][1].extend([c["label"]] * len(X))
    arr = {k: (np.concatenate(v[0]) if v[0] else np.zeros((0, fx.STATIC_DIM)), np.array(v[1])) for k, v in data.items()}
    classes = sorted(set(arr["train"][1]))
    for k in ("val", "test"):
        keep = np.isin(arr[k][1], classes)
        arr[k] = (arr[k][0][keep], arr[k][1][keep])
    log(f"samples: train {len(arr['train'][1])}, val {len(arr['val'][1])}, test {len(arr['test'][1])}; "
        f"classes {len(classes)}; protocol {signers['protocol']}")
    if len(classes) < 2 or len(arr["val"][1]) == 0 or len(arr["test"][1]) == 0:
        raise ValueError("Not enough data: need ≥2 classes and non-empty val/test splits.")

    Xtr, ytr = _balanced(*arr["train"], rng)
    Xva, yva = arr["val"]
    Xte, yte = arr["test"]
    cand, results = candidates(seed), {}
    for name, est in cand.items():
        t0 = time.time()
        est.fit(Xtr, ytr)
        pv = _proba(est, Xva, classes)
        results[name] = {"val_macro_f1": round(float(f1_score(yva, np.array(classes)[pv.argmax(1)],
                                                              average="macro", zero_division=0)), 4),
                         "val_accuracy": round(float((np.array(classes)[pv.argmax(1)] == yva).mean()), 4),
                         "fit_seconds": round(time.time() - t0, 1)}
        pt = _proba(est, Xte, classes)
        results[name]["test_accuracy"] = round(float((np.array(classes)[pt.argmax(1)] == yte).mean()), 4)
        log(f"  {name}: {results[name]}")
    chosen = max(results, key=lambda k: (results[k]["val_macro_f1"], k == "mlp"))
    est = cand[chosen]
    yva_i = np.array([classes.index(v) for v in yva])
    pv = _proba(est, Xva, classes)
    temperature = fit_temperature(pv, yva_i)
    pv_cal = calibrate(pv, temperature)
    threshold, coverage = reject_threshold(pv_cal, yva_i)
    pt = calibrate(_proba(est, Xte, classes), temperature)
    yte_pred = list(np.array(classes)[pt.argmax(1)])
    test = report(list(yte), yte_pred, classes)
    test["top5_accuracy"] = round(float(np.mean([classes.index(t) in np.argsort(-row)[:5]
                                                  for t, row in zip(yte, pt)])), 4)
    test["ece"] = round(ece(pt, np.array([classes.index(t) for t in yte])), 4)

    version = "alphabet-" + time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": chosen, "estimator": est, "classes": classes, "temperature": temperature,
                 "reject_threshold": threshold, "spec_version": fx.SPEC_VERSION, "version": version},
                out / "model.joblib", compress=3)
    (out / "browser.json").write_text(json.dumps(export_browser(est, chosen, classes, temperature, threshold,
                                                                version), separators=(",", ":")))
    metrics = {
        "version": version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "feature_spec": fx.SPEC_VERSION,
        "task": "alphabet_static", "classes": classes, "excluded_dynamic": sorted(DYNAMIC_LETTERS),
        "sources": sources or sorted({c.get("source", "?") for c in clips}),
        "eval_protocol": signers["protocol"], "signers": signers, "warning": warning,
        "samples": {k: int(len(v[1])) for k, v in arr.items()},
        "clips": {k: sum(1 for c in clips if assign[c["clip_id"]] == k) for k in ("train", "val", "test")},
        "candidates": results, "chosen": chosen, "temperature": temperature,
        "reject_threshold": threshold, "val_coverage_at_threshold": round(coverage, 3),
        "val_ece": {"before": round(ece(pv, yva_i), 4), "after": round(ece(pv_cal, yva_i), 4)},
        "test": test, "augmentation": {"copies": n_aug, "rotation_deg": 12, "noise": 0.004},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (Path(out_root) / "CURRENT").write_text(version)
    return metrics, out
