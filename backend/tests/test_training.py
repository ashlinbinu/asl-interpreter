import json
import shutil
import subprocess

import numpy as np
import pytest

from app import features as fx
from app.config import REPO_ROOT
from app.recognition import StaticModel
from app.train_static import make_splits, train
from tests.conftest import synthetic_letter_clips

NODE = shutil.which("node")


def test_splits_never_share_signers():
    clips = synthetic_letter_clips(signers=("a", "b", "c", "d"))
    assign, info, warning = make_splits(clips)
    by_split = {}
    for c in clips:
        by_split.setdefault(assign[c["clip_id"]], set()).add(c["signer_id"])
    assert not (by_split["train"] & by_split["test"]) and not (by_split["val"] & by_split["test"])
    assert info["protocol"] == "signer_holdout" and warning is None


def test_single_signer_is_flagged():
    _, info, warning = make_splits(synthetic_letter_clips(signers=("only",), reps=8))
    assert info["protocol"] == "single_signer_block_split" and "NOT" in warning


def _photos(signer="kaggle_1"):
    clips = synthetic_letter_clips(signers=(signer,), reps=3, seed=5)
    for c in clips:
        c["source"], c["clip_id"] = "kaggle_asl_alphabet", "k-" + c["clip_id"]
    return clips


def test_supplementary_photos_are_train_only():
    clips = synthetic_letter_clips(signers=("a", "b", "c")) + _photos()
    assign, info, _ = make_splits(clips)
    assert all(assign[c["clip_id"]] == "train" for c in clips if c["source"] == "kaggle_asl_alphabet")
    assert info["protocol"] == "signer_holdout" and "kaggle_1" not in info["test"] + info["val"]
    assert info["supplementary_signers"] == ["kaggle_1"]
    with pytest.raises(ValueError, match="training-only"):
        make_splits(clips, test_signers=["kaggle_1"])
    with pytest.raises(ValueError, match="supplementary"):
        make_splits(_photos())


def test_verified_needs_three_video_signers(tmp_path, monkeypatch):
    import app.train_static as ts
    real = ts.candidates
    monkeypatch.setattr(ts, "candidates", lambda seed=0: {"rf": real(seed)["rf"]})
    m3, _ = train(synthetic_letter_clips(), tmp_path / "a", n_aug=0, log=lambda *_: None)
    assert m3["verified"] and set(m3["leave_one_signer_out"]["folds"]) == {"s1", "s2", "s3"}
    assert set(m3["test"]["per_signer"]) == {"s3"}
    m2, out = train(synthetic_letter_clips(signers=("s1", "s2")) + _photos(), tmp_path / "b", n_aug=0,
                    log=lambda *_: None)
    assert not m2["verified"] and m2["supplementary_sources"] == ["kaggle_asl_alphabet"]
    assert m2["ablation"]["without_supplementary"]["n"] > 0
    assert StaticModel(out).summary()["verified"] is False
    crops = synthetic_letter_clips()
    for c in crops:
        c["domain"] = "hand_crop"
    mc, _ = train(crops, tmp_path / "c", n_aug=0, log=lambda *_: None)
    assert not mc["verified"] and mc["test_domain"] == ["hand_crop"] and mc["leave_one_signer_out"]


@pytest.mark.parametrize("kind", ["mlp", "rf"])
def test_train_writes_versioned_artifacts_and_js_matches(tmp_path, kind, monkeypatch):
    import app.train_static as ts
    real = ts.candidates
    monkeypatch.setattr(ts, "candidates", lambda seed=0: {kind: real(seed)[kind]})
    metrics, out = train(synthetic_letter_clips(), tmp_path, n_aug=0, log=lambda *_: None)
    assert (out / "model.joblib").exists() and (out / "metrics.json").exists()
    assert (tmp_path / "CURRENT").read_text() == metrics["version"]
    assert metrics["chosen"] == kind and set(metrics["test"]["per_class"]) == {"A", "B", "C", "H"}
    assert len(metrics["test"]["confusion"]["matrix"]) == 4

    model = StaticModel(out)
    X = np.concatenate([fx.static_vectors(c["frames"])[0] for c in synthetic_letter_clips(signers=("z",), reps=1)])
    py = model.proba(X)
    if NODE is None:
        pytest.skip("node not installed")
    script = (
        "import {readFileSync} from 'node:fs';"
        "import {StaticModel} from './src/lib/staticModel.js';"
        f"const m=new StaticModel(JSON.parse(readFileSync({json.dumps(str(out / 'browser.json'))},'utf8')));"
        f"const X=JSON.parse(readFileSync({json.dumps(str(tmp_path / 'x.json'))},'utf8'));"
        "process.stdout.write(JSON.stringify(X.map(v=>m.proba(v))));"
    )
    (tmp_path / "x.json").write_text(json.dumps(X.tolist()))
    res = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=True,
                         cwd=REPO_ROOT / "frontend")
    js = np.array(json.loads(res.stdout))
    np.testing.assert_allclose(js, py, atol=1e-4 if kind == "mlp" else 0.02)
