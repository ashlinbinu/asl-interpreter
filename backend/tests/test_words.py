"""Word model: torch <-> numpy parity, signer splits, calibrated top-k, component scoring, API endpoints."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app import features as fx  # noqa: E402
from app.word_model import WordModel, WordNet, dtw_path  # noqa: E402
from tests import synth  # noqa: E402

PATHS = {"w-chin": "chin_forward", "w-forehead": "forehead", "w-circle": "chest_circle", "w-still": "still"}


def word_clips(signers=8, reps=3):
    """TEST ONLY: four fake 'words' (different paths/locations) from several fake signers."""
    clips, k = [], 0
    for si in range(signers):
        split = "train" if si < 6 else "val" if si == 6 else "test"
        for label, path in PATHS.items():
            for r in range(reps):
                frames = synth.sign_clip(n=24 + 2 * r, path=path, curl=0.1 * (si % 3), seed=1000 * si + 10 * r + k,
                                         noise=0.003)
                clips.append({"seq": fx.to_sequence(frames), "label": label, "clip_id": k, "signer_id": f"s{si}",
                              "split": split, "source": "test", "frames": frames})
                k += 1
    return clips


@pytest.fixture(scope="module")
def trained_words(tmp_path_factory):
    from app.train_words import train
    clips = word_clips()
    metrics, out = train(clips, tmp_path_factory.mktemp("words"), epochs=30, log=lambda *_: None)
    return metrics, out, clips


def test_splits_and_metrics(trained_words):
    metrics, out, _ = trained_words
    s = metrics["signers"]
    assert s["test"] == ["s7"] and s["val"] == ["s6"] and not set(s["train"]) & {"s6", "s7"}
    assert metrics["eval_protocol"] == "official_signer_split" and not metrics["verified"]   # 1 test signer
    assert metrics["test"]["accuracy"] >= 0.75 and set(metrics["candidates"]) == {"cnn", "bigru"}
    assert 0.0 <= metrics["scoring"]["test_auc"] <= 1.0
    assert (out / "model.json").exists() and (out / "references.npz").exists()


@pytest.mark.parametrize("arch", ["cnn", "bigru"])
def test_numpy_forward_matches_torch(arch):
    from app.train_words import ARCHS
    torch.manual_seed(0)
    net = ARCHS[arch](fx.F, 4).eval()
    if arch == "cnn":   # non-trivial BatchNorm statistics so the folding is exercised
        for b in net.bns:
            b.running_mean.uniform_(-0.5, 0.5)
            b.running_var.uniform_(0.5, 2.0)
    mean, std = np.zeros(fx.F), np.ones(fx.F)
    wn = WordNet({**net.export(), "mean": mean.tolist(), "std": std.tolist()})
    x = np.random.default_rng(0).normal(size=(fx.T, fx.F))
    with torch.no_grad():
        ref = net(torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
    np.testing.assert_allclose(wn.logits(x), ref, atol=1e-4)


def test_dtw_aligns_time_shifted_copies():
    a = np.sin(np.linspace(0, 3, 32))[:, None]
    b = np.sin(np.linspace(0, 3, 32) - 0.3)[:, None]
    _, cost_shift = dtw_path(a, b)
    _, cost_flat = dtw_path(a, np.zeros_like(a))
    assert cost_shift < cost_flat / 3


def test_score_prefers_the_right_sign_and_gives_hints(trained_words):
    _, out, clips = trained_words
    wm = WordModel(out)
    test = [c for c in clips if c["split"] == "test"]
    right, wrong = [], []
    for c in test:
        right.append(wm.score(c["seq"], c["label"])["score"])
        other = next(label for label in PATHS if label != c["label"])
        wrong.append(wm.score(c["seq"], other)["score"])
    assert np.mean(right) > np.mean(wrong) + 20
    r = wm.score(test[0]["seq"], "w-forehead" if test[0]["label"] != "w-forehead" else "w-chin")
    assert set(r["components"]) == {"handshape", "orientation", "location", "movement", "two_hand"}
    assert isinstance(r["hints"], list)


def test_api_word_recognition_and_scoring(client, trained_words, tmp_root, monkeypatch):
    from app.db import Session
    from app.recognition import models
    from app.registry import registry
    from scripts.datasets.common import upsert_word
    _, out, clips = trained_words
    with Session() as s:
        ids = {label: upsert_word(s, label, "test").id for label in PATHS}
        s.commit()
    registry.load()
    monkeypatch.setattr(models, "words", WordModel(out))
    c = next(c for c in clips if c["split"] == "test" and c["label"] == "w-forehead")
    r = client.post("/api/recognize", json={"frames": c["frames"], "mode": "word", "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"].startswith("words_") and len(body["matches"]) == 3
    assert body["matches"][0]["sign_id"] in ids.values()
    r = client.post("/api/score", json={"sign_id": ids["w-forehead"], "frames": c["frames"], "save": False})
    assert r.status_code == 200, r.text
    sc = r.json()
    assert sc["method"] == "words_dtw_components" and 0 <= sc["score"] <= 100 and sc["components"]
    still = client.post("/api/score", json={"sign_id": ids["w-still"], "frames": c["frames"], "save": False}).json()
    assert still["score"] < sc["score"]
