"""Word (dynamic sign) training (docs/PLAN.md P3): 1D-CNN vs BiGRU on FEATURE_SPEC (T=32, F=270) sequences.

Clips: dicts {seq (T, F), label, signer_id, split ("train" | "val" | "test" or None), source}. Splits are by signer:
the dataset's official split when given (ASL Citizen), otherwise assigned by signer. Validation signers pick the
model, fit the temperature, the reject threshold and the score weights; test signers are only used for reporting.
The exported weights run in numpy (app/word_model.py), so torch is needed for training only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from . import features as fx
from .train_static import _split_by_signer, ece, fit_temperature, reject_threshold, report
from .word_model import COMPONENTS, References, WordNet, score_vector, softmax

N_REFS = 5            # reference clips per sign (distinct training signers)
MIN_VERIFIED_SIGNERS = 3


# ---------------------------------------------------------------- networks

class CNN(nn.Module):
    def __init__(self, n_in, n_out, width=128):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(n_in, width, 5, padding=2), nn.Conv1d(width, width, 5, padding=2),
                                    nn.Conv1d(width, width, 3, padding=1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in self.convs])
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * width, n_out)

    def forward(self, x):                       # (B, T, F)
        x = x.transpose(1, 2)
        for c, b in zip(self.convs, self.bns):
            x = tf.relu(b(c(x)))
        return self.out(self.drop(torch.cat([x.mean(2), x.amax(2)], 1)))

    @torch.no_grad()
    def export(self):
        convs = []
        for c, b in zip(self.convs, self.bns):  # fold BatchNorm into the convolution
            s = b.weight / torch.sqrt(b.running_var + b.eps)
            W = c.weight * s[:, None, None]
            convs.append({"W": W.permute(2, 1, 0).tolist(), "b": ((c.bias - b.running_mean) * s + b.bias).tolist(),
                          "pad": int(c.padding[0])})
        return {"kind": "cnn", "convs": convs, "out_W": self.out.weight.T.tolist(), "out_b": self.out.bias.tolist()}


class BiGRU(nn.Module):
    def __init__(self, n_in, n_out, hidden=96, proj=128):
        super().__init__()
        self.proj = nn.Linear(n_in, proj)
        self.gru = nn.GRU(proj, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(4 * hidden, n_out)

    def forward(self, x):
        o, _ = self.gru(tf.relu(self.proj(x)))
        return self.out(self.drop(torch.cat([o.mean(1), o.amax(1)], 1)))

    @torch.no_grad()
    def export(self):
        g = self.gru
        return {"kind": "bigru", "proj_W": self.proj.weight.T.tolist(), "proj_b": self.proj.bias.tolist(),
                "fw": [g.weight_ih_l0.tolist(), g.weight_hh_l0.tolist(), g.bias_ih_l0.tolist(), g.bias_hh_l0.tolist()],
                "bw": [g.weight_ih_l0_reverse.tolist(), g.weight_hh_l0_reverse.tolist(),
                       g.bias_ih_l0_reverse.tolist(), g.bias_hh_l0_reverse.tolist()],
                "out_W": self.out.weight.T.tolist(), "out_b": self.out.bias.tolist()}


ARCHS = {"cnn": CNN, "bigru": BiGRU}


# ---------------------------------------------------------------- data

def assign_splits(clips):
    """Keep given splits; clips without one are split by signer (never mixing a signer across splits)."""
    missing = [c for c in clips if c.get("split") not in ("train", "val", "test")]
    info = {"protocol": "official_signer_split" if not missing else "signer_holdout"}
    if missing:
        assign, sinfo, _ = _split_by_signer(missing)
        for c in missing:
            c["split"] = assign[c["clip_id"]]
        info["assigned"] = sinfo
    for k in ("train", "val", "test"):
        info[k] = sorted({c["signer_id"] for c in clips if c["split"] == k})
    shared = (set(info["train"]) & set(info["test"])) | (set(info["val"]) & set(info["test"]))
    if shared:
        raise ValueError(f"signers in both train/val and test: {sorted(shared)}")
    return info


def augment(x, rng):
    """(B, T, F) standardised -> time warp (speed 0.8-1.25, shift), short time mask, gaussian noise."""
    b, t, _ = x.shape
    out = np.empty_like(x)
    grid = np.arange(t, dtype=np.float64)
    for i in range(b):
        s, sh = rng.uniform(0.8, 1.25), rng.uniform(-2, 2)
        pos = np.clip((grid - t / 2) * s + t / 2 + sh, 0, t - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, t - 1)
        a = (pos - lo)[:, None]
        y = (1 - a) * x[i, lo] + a * x[i, hi]
        if rng.random() < 0.5:
            m = rng.integers(0, t - 3)
            y[m:m + rng.integers(1, 4)] = 0
        out[i] = y + rng.normal(0, 0.05, y.shape)
    return out


def fit(arch, Xtr, ytr, Xva, yva, n_classes, seed=0, epochs=120, patience=20, batch=64, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ARCHS[arch](Xtr.shape[2], n_classes)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, stale = -1.0, None, 0
    Xv = torch.tensor(Xva, dtype=torch.float32)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(Xtr))
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            xb = torch.tensor(augment(Xtr[idx], rng), dtype=torch.float32)
            loss = loss_fn(net(xb), torch.tensor(ytr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xv).argmax(1).numpy()
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        if f1 > best + 1e-4:
            best, stale = f1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    net.load_state_dict(best_state)
    net.eval()
    log(f"  {arch}: best val macro-F1 {best:.3f} after {ep + 1} epochs")
    return net, best, ep + 1


# ---------------------------------------------------------------- references + score calibration

def pick_references(clips, net: WordNet, k=N_REFS):
    """Per class, up to k training clips from distinct signers, closest to the class mean (typical productions)."""
    out = []
    for label in sorted({c["label"] for c in clips}):
        own = [c for c in clips if c["label"] == label]
        X = np.stack([net.standardize(c["seq"]) for c in own])
        d = np.sqrt(((X - X.mean(0)) ** 2).sum((1, 2)))
        seen = set()
        for i in np.argsort(d):
            if own[i]["signer_id"] not in seen:
                seen.add(own[i]["signer_id"])
                out.append(own[i])
            if len(seen) == k:
                break
    return out


def component_stats(train_clips, refs: References, net: WordNet, max_per_class=60, seed=0):
    """Typical same-sign distances: other training signers' clips vs. their sign's references."""
    rng = np.random.default_rng(seed)
    stats = {}
    ref_ids = {(s, l) for s, l in zip(refs.signers, refs.labels)}
    for label in sorted(set(refs.labels)):
        own = [c for c in train_clips if c["label"] == label and (c["signer_id"], label) not in ref_ids]
        if len(own) > max_per_class:
            own = [own[i] for i in rng.choice(len(own), max_per_class, replace=False)]
        dists = [refs.compare(net.standardize(c["seq"]), label)[1] for c in own]
        stats[label] = {g: sorted(round(d[g], 5) for d in dists) for g in COMPONENTS}
    return stats


def score_features(net, refs, temperature, classes, clips, rng, negatives=3):
    """For each clip: a positive pair (its own sign) and `negatives` pairs with random other signs."""
    X, y = [], []
    for c in clips:
        p = softmax(net.logits(c["seq"]), temperature)
        x = net.standardize(c["seq"])
        others = [k for k in classes if k != c["label"]]
        pairs = [(c["label"], 1)] + [(others[i], 0) for i in rng.choice(len(others), min(negatives, len(others)),
                                                                          replace=False)]
        for label, target in pairs:
            pct = refs.percentiles(label, refs.compare(x, label)[1])
            X.append(score_vector(float(p[classes.index(label)]), pct))
            y.append(target)
    return np.array(X), np.array(y)


# ---------------------------------------------------------------- main entry

def train(clips, out_root: Path, archs=("cnn", "bigru"), seed=0, epochs=120, log=print):
    t0 = time.time()
    clips = [c for c in clips if c.get("seq") is not None]
    signers = assign_splits(clips)
    parts = {k: [c for c in clips if c["split"] == k] for k in ("train", "val", "test")}
    classes = sorted({c["label"] for c in parts["train"]})
    for k in ("val", "test"):
        parts[k] = [c for c in parts[k] if c["label"] in classes]
    if len(classes) < 2 or not parts["val"] or not parts["test"]:
        raise ValueError("Need >= 2 classes and non-empty validation and test signer sets.")
    log(f"clips: train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])}; "
        f"{len(classes)} classes; signers {len(signers['train'])}/{len(signers['val'])}/{len(signers['test'])}")

    raw_tr = np.stack([c["seq"] for c in parts["train"]])
    mean = raw_tr.reshape(-1, raw_tr.shape[2]).mean(0)
    std = np.maximum(raw_tr.reshape(-1, raw_tr.shape[2]).std(0), 1e-3)
    norm = lambda cs: np.stack([(c["seq"] - mean) / std for c in cs]).astype(np.float32)  # noqa: E731
    lab = lambda cs: np.array([classes.index(c["label"]) for c in cs])  # noqa: E731
    Xtr, Xva, Xte = norm(parts["train"]), norm(parts["val"]), norm(parts["test"])
    ytr, yva, yte = lab(parts["train"]), lab(parts["val"]), lab(parts["test"])

    results, nets = {}, {}
    for arch in archs:
        t = time.time()
        net, f1, epochs_run = fit(arch, Xtr, ytr, Xva, yva, len(classes), seed, epochs, log=log)
        exported = {**net.export(), "mean": mean.tolist(), "std": std.tolist()}
        wn = WordNet(exported)
        pt = softmax(np.stack([wn.logits(c["seq"]) for c in parts["test"]]))
        results[arch] = {"val_macro_f1": round(float(f1), 4), "epochs": epochs_run,
                         "test_accuracy": round(float((pt.argmax(1) == yte).mean()), 4),
                         "fit_seconds": round(time.time() - t, 1)}
        nets[arch] = (wn, exported)
    chosen = max(results, key=lambda k: (results[k]["val_macro_f1"], k == "cnn"))
    wn, exported = nets[chosen]

    lv = np.stack([wn.logits(c["seq"]) for c in parts["val"]])
    pv_raw = softmax(lv)
    temperature = fit_temperature(pv_raw, yva)
    pv = softmax(lv, temperature)
    threshold, coverage = reject_threshold(pv, yva)
    pt = softmax(np.stack([wn.logits(c["seq"]) for c in parts["test"]]), temperature)
    pred = [classes[i] for i in pt.argmax(1)]
    truth = [c["label"] for c in parts["test"]]
    test = report(truth, pred, classes)
    test["top5_accuracy"] = round(float(np.mean([yte[i] in np.argsort(-pt[i])[:5] for i in range(len(yte))])), 4)
    test["ece"] = round(ece(pt, yte), 4)
    sig = np.array([c["signer_id"] for c in parts["test"]])
    ok = pt.argmax(1) == yte
    test["per_signer"] = {s: {"accuracy": round(float(ok[sig == s].mean()), 4), "n": int((sig == s).sum())}
                          for s in sorted(set(sig))}

    log("references + component statistics ...")
    ref_clips = pick_references(parts["train"], wn)
    refs = References(np.stack([c["seq"] for c in ref_clips]).astype(np.float32), [c["label"] for c in ref_clips],
                      [c["signer_id"] for c in ref_clips], {}, wn)
    refs.stats = component_stats(parts["train"], refs, wn, seed=seed)

    log("score calibration on validation signers ...")
    rng = np.random.default_rng(seed)
    Xs_va, ys_va = score_features(wn, refs, temperature, classes, parts["val"], rng)
    lr = LogisticRegression(C=0.3, max_iter=1000).fit(Xs_va, ys_va)
    Xs_te, ys_te = score_features(wn, refs, temperature, classes, parts["test"], rng)
    auc_te = roc_auc_score(ys_te, lr.predict_proba(Xs_te)[:, 1])
    auc_prob_only = roc_auc_score(ys_te, Xs_te[:, 0])
    comp_auc = {g: round(float(roc_auc_score(ys_te, Xs_te[:, 1 + k])), 4) for k, g in enumerate(COMPONENTS)
                if len(set(Xs_te[:, 1 + k])) > 1}
    weights = {"features": ["target_logodds"] + list(COMPONENTS), "coef": lr.coef_[0].tolist(),
               "intercept": float(lr.intercept_[0])}

    version = "words-" + time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / version
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "references.npz", seqs=refs.seqs, labels=np.array(refs.labels),
                        signers=np.array(refs.signers))
    (out / "model.json").write_text(json.dumps({
        "version": version, "spec_version": fx.SPEC_VERSION, "classes": classes, "net": exported,
        "temperature": temperature, "reject_threshold": threshold, "score_weights": weights,
        "component_stats": refs.stats}, separators=(",", ":")))
    metrics = {
        "version": version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "feature_spec": fx.SPEC_VERSION,
        "task": "words_dynamic", "classes": classes, "sources": sorted({c.get("source", "?") for c in clips}),
        "eval_protocol": signers["protocol"], "signers": {k: signers[k] for k in ("train", "val", "test")},
        "verified": len(signers["test"]) >= MIN_VERIFIED_SIGNERS,
        "clips": {k: len(v) for k, v in parts.items()}, "candidates": results, "chosen": chosen,
        "temperature": temperature, "reject_threshold": threshold, "val_coverage_at_threshold": round(coverage, 3),
        "val_ece": {"before": round(ece(pv_raw, yva), 4), "after": round(ece(pv, yva), 4)}, "test": test,
        "scoring": {"method": "logistic(target probability + DTW component percentiles), fitted on val signers",
                    "test_auc": round(float(auc_te), 4), "test_auc_probability_only": round(float(auc_prob_only), 4),
                    "test_auc_per_component": comp_auc, "pairs_test": int(len(ys_te)), "weights": weights},
        "references": {"per_sign": N_REFS, "from": "training signers only"},
        "augmentation": "time warp 0.8-1.25x, shift +-2, time mask <=3 frames, noise 0.05 (standardised)",
        "seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (Path(out_root) / "CURRENT").write_text(version)
    return metrics, out
