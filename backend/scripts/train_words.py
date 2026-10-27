"""Train the word (dynamic sign) model from every word reference clip in the DB (real signer data only).

    python -m scripts.train_words                    # CNN vs BiGRU, official signer splits where available
    python -m scripts.train_words --archs cnn --epochs 80

Feature sequences are cached in <store>/cache/word_sequences_<spec>.npz. Writes
$HANDSPEAK_MODELS_DIR/words/<version>/{model.json, references.npz, metrics.json} and updates CURRENT.
Restart the API afterwards. Needs torch (training only): see requirements-train.txt.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import features as fx  # noqa: E402
from app.clipstore import load_clip  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import ReferenceClip, Session, Sign, init_db  # noqa: E402


def load_clips(sources=None, log=print):
    st = get_settings()
    cache_path = st.store_dir / "cache" / f"word_sequences_{fx.SPEC_VERSION}.npz"
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["seqs"]))
    with Session() as s:
        q = s.query(ReferenceClip, Sign).join(Sign, Sign.id == ReferenceClip.sign_id).filter(Sign.category == "word")
        if sources:
            q = q.filter(ReferenceClip.source.in_(sources))
        rows = q.all()
    clips, added, skipped = [], 0, 0
    for clip, sign in rows:
        key = clip.landmarks_path
        if key not in cache:
            path = st.store_dir / key
            if not path.exists():
                skipped += 1
                continue
            frames, meta = load_clip(path)
            seq = fx.to_sequence(frames, bool((clip.quality or {}).get("left_handed") or meta.get("left_handed")))
            if seq is None:
                skipped += 1
                continue
            cache[key] = seq.astype(np.float32)
            added += 1
        clips.append({"seq": cache[key].astype(np.float64), "label": sign.label, "clip_id": clip.id,
                      "signer_id": f"{clip.source}:{clip.signer_id}", "split": clip.split, "source": clip.source})
    if added:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(cache)
        np.savez_compressed(cache_path, keys=np.array(keys), seqs=np.stack([cache[k] for k in keys]))
    log(f"{len(clips)} clips ({added} newly featurised, {skipped} skipped)")
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--archs", nargs="*", default=["cnn", "bigru"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from app.train_words import train  # torch import only when training
    init_db()
    clips = load_clips(a.sources)
    if not clips:
        raise SystemExit("No word clips found. Run scripts.datasets.ingest_asl_citizen first (see README).")
    metrics, out = train(clips, get_settings().models_dir / "words", archs=a.archs, seed=a.seed, epochs=a.epochs)
    t = metrics["test"]
    print(json.dumps({k: metrics[k] for k in ("version", "eval_protocol", "chosen", "candidates", "temperature",
                                              "reject_threshold", "clips")}, indent=2))
    print(f"TEST (held-out signers: {len(metrics['signers']['test'])}) top-1 {t['accuracy']:.3f}  "
          f"top-5 {t['top5_accuracy']:.3f}  macro-F1 {t['macro_f1']:.3f}  ECE {t['ece']:.3f}  (n={t['n']})")
    for c in t["top_confusions"][:8]:
        print(f"  {c['true']} -> {c['pred']}: {c['rate']:.0%} ({c['count']})")
    sc = metrics["scoring"]
    print(f"Scoring AUC (correct vs other sign, test signers): {sc['test_auc']:.3f} "
          f"(probability alone {sc['test_auc_probability_only']:.3f}); per component {sc['test_auc_per_component']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
