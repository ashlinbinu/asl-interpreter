"""Calibrate scoring against a sign's *own* recordings (custom signs, self-recorded letters) on real ASL Citizen data.

    python -m scripts.calibrate_own_clip          # writes own_clip_calibration.json next to the current word model

When you practise a custom sign, your attempt is compared with your own recording: the same person, so the right
yardstick is the same person making a *different* sign, not other signers. From ASL Citizen clips (word model
standardisation, banded DTW, per-component distances):
  positives = the same signer repeating the same word, negatives = the same signer making a different word.
Each component distance becomes "share of same-person different-sign distances at least this large" (training
signers). The score must never rise when a component gets worse, so only monotone combinations are considered:
a logistic regression on all five with non-negative weights, or their plain mean through a one-input logistic
curve. They are compared by signer-grouped cross-validation on the fitting signers only; the winner is refitted
on train + validation signers and evaluated once on the held-out test signers (ROC AUC). Scores are divided by
the model's value for a perfect match (every component closer than all different signs), so a perfect match is
100; this is monotone, so the AUC is unchanged. This measures the method on real signers'
repeats; it is not a measurement on your own custom signs.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from app.recognition import current_dir  # noqa: E402
from app.word_model import COMPONENTS, WordModel, component_distances, dtw_path  # noqa: E402
from scripts.train_words import load_clips  # noqa: E402

QUANTILES = 101
NEG_PER_SIGNER = 60


class MonotoneLogistic:
    """Class-balanced, L2-regularised logistic regression whose weights are constrained to be >= 0."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        w_pos, w_neg = len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max((1 - y).sum(), 1))
        sw = np.where(y == 1, w_pos, w_neg)

        def loss(p):
            z = X @ p[:-1] + p[-1]
            nll = sw * (np.logaddexp(0, z) - y * z)
            grad_z = sw * (1 / (1 + np.exp(-z)) - y)
            return (0.5 * p[:-1] @ p[:-1] + self.C * nll.sum(),
                    np.append(p[:-1] + self.C * X.T @ grad_z, self.C * grad_z.sum()))

        k = X.shape[1]
        r = minimize(loss, np.zeros(k + 1), jac=True, method="L-BFGS-B", bounds=[(0, None)] * k + [(None, None)])
        self.coef_, self.intercept_ = r.x[:-1], r.x[-1]
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X @ self.coef_ + self.intercept_)))
        return np.stack([1 - p, p], axis=1)


def survival(qs: dict, dists: dict) -> list[float]:
    """Per component: share of reference (negative) distances >= d, from stored quantiles (0-1)."""
    return [float(1 - np.searchsorted(qs[g], dists[g], side="left") / (len(qs[g]) - 1)) if qs.get(g) else 0.0
            for g in COMPONENTS]


def pairs(clips, wm, rng):
    """Same-signer pairs -> {split: [(distances, is_same_sign, signer)]}."""
    by_signer = collections.defaultdict(list)
    for c in clips:
        by_signer[c["signer_id"]].append(c)
    out = collections.defaultdict(list)

    def dist(a, b):
        return component_distances(a["std"], b["std"], dtw_path(a["std"], b["std"])[0])

    for signer, cs in by_signer.items():
        split = cs[0]["split"]
        by_label = collections.defaultdict(list)
        for c in cs:
            by_label[c["label"]].append(c)
        for group in by_label.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    out[split].append((dist(group[i], group[j]), 1, signer))
        if len(cs) >= 2:
            for _ in range(NEG_PER_SIGNER):
                a, b = rng.choice(len(cs), 2, replace=False)
                if cs[a]["label"] != cs[b]["label"]:
                    out[split].append((dist(cs[a], cs[b]), 0, signer))
    return out


def calibrate(wm: WordModel, clips, seed=0, log=print) -> dict:
    rng = np.random.default_rng(seed)
    for c in clips:
        c["std"] = wm.net.standardize(c["seq"])
    P = pairs(clips, wm, rng)
    fit = P["train"] + P["val"]
    test = P["test"]
    neg_train = [d for d, y, _ in P["train"] if y == 0]
    qs = {g: np.quantile([d[g] for d in neg_train], np.linspace(0, 1, QUANTILES)).round(5).tolist() for g in COMPONENTS}
    X = np.array([survival(qs, d) for d, _, _ in fit])
    y = np.array([y for _, y, _ in fit])
    groups = np.array([s for _, _, s in fit])
    inputs = {"all_components": lambda A: A, "mean_of_components": lambda A: A.mean(1, keepdims=True)}
    cv = {}
    for name, f in inputs.items():
        scores = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = MonotoneLogistic().fit(f(X[tr]), y[tr])
            scores[te] = m.predict_proba(f(X[te]))[:, 1]
        cv[name] = round(float(roc_auc_score(y, scores)), 4)
    combine = max(cv, key=cv.get)
    lr = MonotoneLogistic().fit(inputs[combine](X), y)
    p_perfect = float(lr.predict_proba(inputs[combine](np.ones((1, len(COMPONENTS)))))[0, 1])
    Xt = np.array([survival(qs, d) for d, _, _ in test])
    yt = np.array([y for _, y, _ in test])
    st = np.minimum(lr.predict_proba(inputs[combine](Xt))[:, 1] / p_perfect, 1.0)
    metrics = {
        "combine": combine, "cv_auc_on_fit_signers": cv,
        "eval_protocol": "ASL Citizen test signers (held out of the fit): same signer repeating a word vs the same "
                         "signer making a different word",
        "test_auc": round(float(roc_auc_score(yt, st)), 4),
        "test_auc_per_component": {g: round(float(roc_auc_score(yt, Xt[:, k])), 4) for k, g in enumerate(COMPONENTS)},
        "test_pairs": {"same_sign": int(yt.sum()), "different_sign": int((1 - yt).sum())},
        "fit_pairs": {"same_sign": int(y.sum()), "different_sign": int((1 - y).sum())},
        "test_score_median": {"same_sign": round(float(np.median(st[yt == 1]) * 100), 1),
                              "different_sign": round(float(np.median(st[yt == 0]) * 100), 1)},
    }
    log(f"own-recording scoring AUC on test signers {metrics['test_auc']} "
        f"({metrics['test_pairs']['same_sign']} repeat / {metrics['test_pairs']['different_sign']} different-sign "
        f"pairs); median score {metrics['test_score_median']}")
    return {"version": wm.version, "components": list(COMPONENTS), "neg_quantiles": qs, "combine": combine,
            "coef": np.round(lr.coef_, 5).tolist(), "intercept": round(float(lr.intercept_), 5),
            "perfect_probability": round(p_perfect, 5),
            "metrics": metrics, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = current_dir("words")
    if d is None:
        ap.error("no trained word model: run python -m scripts.train_words first")
    cal = calibrate(WordModel(d), [c for c in load_clips() if c["source"] == "asl_citizen"], a.seed)
    out = d / "own_clip_calibration.json"
    out.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(f"Wrote {out}. Restart the API to use it.")


if __name__ == "__main__":
    main()
