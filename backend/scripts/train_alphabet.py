"""Train the static alphabet classifier from every alphabet reference clip in the DB (real landmarks only).

    python -m scripts.train_alphabet                       # all sources, signer-held-out split
    python -m scripts.train_alphabet --test-signers me_01  # report on your own recordings
    python -m scripts.train_alphabet --sources own kaggle_asl_alphabet --stride 3

Writes $HANDSPEAK_MODELS_DIR/alphabet/<version>/{model.joblib, browser.json, metrics.json} and updates CURRENT.
Restart the API afterwards.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clipstore import load_clip  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import ReferenceClip, Session, Sign, init_db  # noqa: E402
from app.train_static import train  # noqa: E402


def load_clips(sources=None, min_hand_rate=0.5):
    store = get_settings().store_dir
    clips = []
    with Session() as s:
        q = (s.query(ReferenceClip, Sign).join(Sign, Sign.id == ReferenceClip.sign_id)
             .filter(Sign.category == "alphabet", Sign.kind == "static"))
        if sources:
            q = q.filter(ReferenceClip.source.in_(sources))
        rows = q.all()
    for clip, sign in rows:
        path = store / clip.landmarks_path
        if not path.exists():
            continue
        if (clip.quality or {}).get("hand_rate", 1.0) < min_hand_rate:
            continue
        frames, meta = load_clip(path)
        clips.append({"frames": frames, "label": sign.label, "signer_id": f"{clip.source}:{clip.signer_id}",
                      "clip_id": clip.id, "source": clip.source,
                      "supplementary": bool((clip.quality or {}).get("supplementary")),
                      "left_handed": bool((clip.quality or {}).get("left_handed") or meta.get("left_handed"))})
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="*", help="restrict to these sources (default: all)")
    ap.add_argument("--test-signers", nargs="*", help="signer ids for the test split, as source:signer_id")
    ap.add_argument("--val-signers", nargs="*")
    ap.add_argument("--augment", type=int, default=2, help="augmented copies per training sample")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth frame of video clips")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    init_db()
    clips = load_clips(a.sources)
    signers = sorted({c["signer_id"] for c in clips})
    print(f"{len(clips)} clips from {len(signers)} signer(s): {', '.join(signers)}")
    if not clips:
        raise SystemExit("No alphabet clips found. Ingest images or record yourself first (see README).")
    metrics, out = train(clips, get_settings().models_dir / "alphabet", sources=a.sources,
                         test_signers=a.test_signers, val_signers=a.val_signers, seed=a.seed, n_aug=a.augment,
                         stride=a.stride)
    t = metrics["test"]
    print(json.dumps({k: metrics[k] for k in ("version", "eval_protocol", "signers", "chosen", "candidates",
                                              "temperature", "reject_threshold")}, indent=2))
    print(f"TEST accuracy {t['accuracy']:.3f}  macro-F1 {t['macro_f1']:.3f}  top-5 {t['top5_accuracy']:.3f}  "
          f"ECE {t['ece']:.3f}  (n={t['n']})")
    for c in t["top_confusions"][:8]:
        print(f"  {c['true']} -> {c['pred']}: {c['rate']:.0%} ({c['count']})")
    cv = metrics.get("leave_one_signer_out")
    if cv:
        print(f"Leave-one-signer-out ({cv['model']}): mean {cv['mean']:.3f}, range {cv['min']:.3f}-{cv['max']:.3f}")
    if metrics.get("ablation"):
        print("Without supplementary photos:", metrics["ablation"]["without_supplementary"])
    if not metrics["verified"]:
        print(f"UNVERIFIED: needs >= 3 held-out video signers (has {len(metrics['video_signers'])}).")
    if metrics.get("warning"):
        print("WARNING:", metrics["warning"])
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
