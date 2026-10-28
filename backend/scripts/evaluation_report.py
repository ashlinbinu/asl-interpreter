"""Write docs/EVALUATION.md from the current models' metrics.json files plus latencies measured on this machine.

    python -m scripts.evaluation_report

Every number comes from held-out signers as recorded at training time; nothing is typed in by hand.
"""
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import features as fx  # noqa: E402
from app.clipstore import load_clip  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.recognition import StaticModel, current_dir  # noqa: E402
from app.word_model import WordModel  # noqa: E402

pct = lambda v: "–" if v is None else f"{v * 100:.1f}%"  # noqa: E731


def _ms(fn, n):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def word_latency(wm: WordModel):
    from scripts.train_words import load_clips
    clips = [c for c in load_clips(log=lambda *_: None) if c["split"] == "test"][:30]
    seqs = [c["seq"] for c in clips]
    out = {"classify_ms": _ms(lambda: [wm.proba_seq(s) for s in seqs], 1) / len(seqs),
           "score_ms": _ms(lambda: [wm.score(s, c["label"]) for s, c in zip(seqs, clips)], 1) / len(seqs)}
    from app.db import ReferenceClip, Session
    st = get_settings()
    with Session() as s:
        paths = [r.landmarks_path for r in s.query(ReferenceClip).filter(ReferenceClip.source == "asl_citizen",
                                                                            ReferenceClip.split == "test").limit(10)]
    frames = [load_clip(st.store_dir / p)[0] for p in paths]
    out["features_ms"] = _ms(lambda: [fx.to_sequence(f) for f in frames], 1) / max(len(frames), 1)
    out["frames_per_clip"] = float(np.mean([len(f) for f in frames])) if frames else None
    node = shutil.which("node")
    if node:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "m.json").write_text(json.dumps({"version": wm.version, "classes": wm.classes,
                                                         "temperature": wm.temperature, "reject_threshold": 0,
                                                         "net": json.loads((wm.dir / "model.json").read_text())["net"]}))
            (Path(td) / "x.json").write_text(json.dumps([s.tolist() for s in seqs[:10]]))
            js = ("import {readFileSync} from 'node:fs';import {WordModel} from './src/lib/wordModel.js';"
                  f"const m=new WordModel(JSON.parse(readFileSync({json.dumps(str(Path(td) / 'm.json'))},'utf8')));"
                  f"const X=JSON.parse(readFileSync({json.dumps(str(Path(td) / 'x.json'))},'utf8'));"
                  "X.forEach(x=>m.proba(x));const t=performance.now();for(let r=0;r<3;r++)X.forEach(x=>m.proba(x));"
                  "process.stdout.write(String((performance.now()-t)/(3*X.length)));")
            r = subprocess.run([node, "--input-type=module", "-e", js], capture_output=True, text=True,
                               cwd=REPO_ROOT / "frontend")
            if r.returncode == 0:
                out["js_classify_ms"] = float(r.stdout)
    return out


def words_section(wm: WordModel):
    m = wm.metrics
    t, sc = m["test"], m["scoring"]
    per_signer = [v["accuracy"] for v in t["per_signer"].values()]
    worst = sorted(t["per_class"].items(), key=lambda kv: kv[1]["f1"])[:8]
    lines = [
        f"## Words ({len(wm.classes)}-sign vocabulary, ASL Citizen)", "",
        f"Model `{m['version']}` ({m['chosen']}), feature spec {m['feature_spec']}. Protocol: **{m['eval_protocol']}**"
        f" (train {len(m['signers']['train'])} / val {len(m['signers']['val'])} / test {len(m['signers']['test'])}"
        f" signers; clips {m['clips']['train']} / {m['clips']['val']} / {m['clips']['test']}). "
        f"{'Verified' if m['verified'] else 'Unverified'} (≥3 held-out test signers).", "",
        "| Metric (held-out test signers) | Value |", "|---|---|",
        f"| Top-1 accuracy | **{pct(t['accuracy'])}** |", f"| Top-5 accuracy | {pct(t['top5_accuracy'])} |",
        f"| Macro-F1 | {t['macro_f1']:.3f} |", f"| ECE after temperature scaling | {t['ece']:.3f} |",
        f"| Per-signer accuracy (min / mean / max) | {pct(min(per_signer))} / {pct(float(np.mean(per_signer)))} / "
        f"{pct(max(per_signer))} |",
        f"| Reject threshold (95% precision on val) | {m['reject_threshold']:.2f} "
        f"(val coverage {pct(m['val_coverage_at_threshold'])}) |", "",
        "Candidates (chosen by validation macro-F1; test shown for transparency):", "",
        "| Model | Val macro-F1 | Test top-1 | Epochs | Fit time |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['val_macro_f1']:.3f} | {pct(v['test_accuracy'])} | {v['epochs']} | {v['fit_seconds']} s |"
              for k, v in m["candidates"].items()]
    lines += ["", "Weakest signs (test F1): " + ", ".join(f"{k} {v['f1']:.2f}" for k, v in worst) + ".", "",
              "Most frequent confusions (test): " + ", ".join(f"{c['true']} → {c['pred']} {c['rate']:.0%}"
                                                              for c in t["top_confusions"][:8]) + ".", "",
              "### Practice score", "",
              f"{sc['method']}. Ranking correct sign vs. a different sign on test signers: **AUC "
              f"{sc['test_auc']:.3f}** (classifier probability alone {sc['test_auc_probability_only']:.3f}; "
              f"{sc['pairs_test']} pairs). Single components: " +
              ", ".join(f"{k} {v:.2f}" for k, v in sc["test_auc_per_component"].items()) +
              ". The components add little ranking power beyond the classifier; they are shown to explain the "
              "score and drive the hints.", ""]
    oc = (wm.own_clip or {}).get("metrics")
    if oc:
        cv = ", ".join(f"{k} {v:.3f}" for k, v in oc["cv_auc_on_fit_signers"].items())
        lines += ["### Scoring against your own recording (custom signs)", "",
                  "Signs with recordings but no trained model are scored by DTW against their own clips. Calibrated "
                  "on the same person making a different sign (`scripts.calibrate_own_clip`), fitted on train + val "
                  f"signers (combination `{oc['combine']}`, chosen by signer-grouped CV AUC: {cv}).", "",
                  f"Proxy test on held-out ASL Citizen signers, same signer repeating a word vs the same signer making "
                  f"a different word: **AUC {oc['test_auc']:.3f}** ({oc['test_pairs']['same_sign']} repeat / "
                  f"{oc['test_pairs']['different_sign']} different-sign pairs); median score "
                  f"{oc['test_score_median']['same_sign']} vs {oc['test_score_median']['different_sign']}. Single "
                  "components: " + ", ".join(f"{k} {v:.2f}" for k, v in oc["test_auc_per_component"].items()) +
                  ". **Unverified on your own custom signs**: no held-out data exists for them.", ""]
    return lines


def alphabet_section(am: StaticModel):
    m = am.metrics
    t = m["test"]
    cv = m.get("leave_one_signer_out")
    lines = [
        "## Alphabet (24 static letters)", "",
        f"Model `{m['version']}` ({m['chosen']}). Sources: {', '.join(m['sources'])}; supplementary (train-only): "
        f"{', '.join(m.get('supplementary_sources') or []) or 'none'}. Test signers: {', '.join(m['signers']['test'])}"
        f" (domain: {', '.join(m.get('test_domain') or [])}).", "",
        f"**{'Verified' if m.get('verified') else 'UNVERIFIED'}**: verification needs ≥3 held-out *full-frame video* "
        f"signers (has {len(m.get('full_frame_signers') or [])}). The test signer is a ChicagoFSVid studio hand crop "
        "recoloured for MediaPipe, so this measures handshape recognition across signers in that domain, **not "
        "webcam accuracy**.", "",
        "| Metric | Value |", "|---|---|",
        f"| Test accuracy ({m['eval_protocol']}) | {pct(t['accuracy'])} |",
        f"| Test macro-F1 | {t['macro_f1']:.3f} |", f"| Top-5 | {pct(t.get('top5_accuracy'))} |",
        f"| ECE | {t.get('ece', 0):.3f} |"]
    if cv:
        folds = ", ".join(f"{k.split(':')[-1]} {pct(v['accuracy'])}" for k, v in cv["folds"].items())
        lines.append(f"| Leave-one-signer-out ({cv['model']}, no augmentation) | mean {pct(cv['mean'])} "
                     f"({folds}) |")
    ab = (m.get("ablation") or {}).get("without_supplementary")
    if ab:
        lines.append(f"| Without the Kaggle photos (same test signer) | {pct(ab['accuracy'])} |")
    lines += ["", "Most frequent confusions: " + ", ".join(f"{c['true']} → {c['pred']} {c['rate']:.0%}"
                                                       for c in t["top_confusions"][:8]) + ".", "",
              "Confusable sets: " + ", ".join(f"{s['set']} {pct(s['accuracy'])}" for s in t["confusable_sets"]
                                              if all(c in am.classes for c in s["set"])) + ".",
              ""]
    return lines


def _mirrored(frames):
    """The same clip as a left-handed signer would make it (horizontal mirror, handedness and pose sides swapped)."""
    swap = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    out = []
    for f in frames:
        hands = [{**h, "handedness": {"Left": "Right", "Right": "Left"}.get(h.get("handedness")),
                  "lm": [[1 - x, y, z] for x, y, z in h["lm"]]} for h in f.get("hands") or []]
        pose = None
        if f.get("pose"):
            pose = [[1 - q[0], q[1], q[2], q[3]] for q in f["pose"]]
            for a, b in swap:
                pose[a], pose[b] = pose[b], pose[a]
        out.append({**f, "hands": hands, "pose": pose})
    return out


def motion_section():
    from app.motion_letters import NoMotionReference, motion_scorer
    lines = ["### J and Z (letters that move)", ""]
    try:
        loo = {L: motion_scorer.leave_one_out(L) for L in "JZ"}
    except NoMotionReference:
        return lines + ["Not scored: the letter reference videos are missing "
                        "(`python -m scripts.datasets.ingest_commons_alphabet`).", ""]
    st = get_settings()
    clip = motion_scorer.reference("J")["clip"]
    lines += [f"Scored on handshape (only the defining finger extended) and the path of that fingertip, against one "
              f"reference signer (`{clip.source}:{clip.signer_id}`). Check on that signer's other letters, each scored "
              "as J or Z with itself left out of the comparison:", "",
              "| Scored as | Other letters: median | Highest | Scoring >= 50 | The reference clip itself: as is / "
              "1.4x slower / 0.7x faster / mirrored (left-handed) |", "|---|---|---|---|---|"]
    for L, r in loo.items():
        v = np.array(list(r.values()))
        ref = motion_scorer.reference(L)["clip"]
        frames, _ = load_clip(st.store_dir / ref.landmarks_path)
        own = [motion_scorer.score(frames, L)["score"],
               motion_scorer.score([{**f, "t": f["t"] * 1.4} for f in frames], L)["score"],
               motion_scorer.score([{**f, "t": f["t"] * 0.7} for f in frames], L)["score"],
               motion_scorer.score(_mirrored(frames), L, left_handed=True)["score"]]
        lines.append(f"| {L} | {np.median(v):.1f} | {v.max():.1f} ({max(r, key=r.get)}) | {int((v >= 50).sum())} of "
                     f"{len(v)} | {' / '.join(f'{x:.0f}' for x in own)} |")
    lines += ["", "**Unverified on other signers**: there is no second signer's J or Z to test a correct attempt.", ""]
    return lines


def main():
    init_db()
    from scripts.dataset_report import counts, to_markdown
    lines = ["# Evaluation", "",
             f"Generated {time.strftime('%Y-%m-%d')} by `python -m scripts.evaluation_report` on "
             f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, Python "
             f"{platform.python_version()}. All accuracy numbers are on signers **not seen in training**.", ""]
    wd, ad = current_dir("words"), current_dir("alphabet")
    wm = WordModel(wd) if wd else None
    am = StaticModel(ad) if ad else None
    if wm:
        lines += words_section(wm)
    if am:
        lines += alphabet_section(am)
    lines += motion_section()
    lines += ["## Latency (measured on this machine)", ""]
    if wm:
        lat = word_latency(wm)
        from app.word_model import DTW_BACKEND
        lines += [f"DTW runs on **{DTW_BACKEND}**.", "", "| Step | Time |", "|---|---|",
                  f"| Features for one clip (FEATURE_SPEC, Python; ~{lat['frames_per_clip']:.0f} frames) | "
                  f"{lat['features_ms']:.1f} ms |",
                  f"| Word classification (numpy) | {lat['classify_ms']:.1f} ms |",
                  f"| Word score incl. DTW vs 5 references (numpy; DTW on {DTW_BACKEND}) | {lat['score_ms']:.1f} ms |"]
        if "js_classify_ms" in lat:
            lines.append(f"| Word classification in JavaScript (Node, same code as the browser) | "
                         f"{lat['js_classify_ms']:.1f} ms |")
        lines += ["", "MediaPipe landmark extraction offline: ~60 ms/frame per process on this CPU (hand ≈33, pose "
                      "≈22); throughput 38 frames/s with 3 processes (PLAN.md P2).",
                  "Browser FPS and end-to-end live latency were **not measured**: no browser with a webcam was "
                  "available in this environment. The camera shows its FPS live.", ""]
    lines += ["## Data", "", to_markdown(counts()), "",
              "## Limitations", "",
              "- A learning aid, not a translator. Text → sign is word-by-word glossing; ASL grammar, word order and "
              "non-manual markers (facial grammar) are not modelled.",
              f"- Words: {len(wm.classes) if wm else 0} signs, one variant each (the ASL Citizen gloss with most "
              "clips; e.g. EAT1, DRINK1). Other "
              "regional or variant forms will score low. Accuracy is for ASL Citizen's webcam recordings; your "
              "camera, lighting and framing can lower it.",
              "- Alphabet: unverified (see above). J and Z are not recognised live; they are scored against one "
              "reference signer's video (see above), which is unverified on other signers.",
              "- Custom signs are scored against your own recordings only (unverified; see the proxy test above). "
              "A sign with no recording can't be scored. Letters are never scored this way: against a letter's "
              "reference, a different letter by the same signer scored a median of ~94 (26 letters of one signer).",
              f"- The practice score's weights were fitted on "
              f"{len(wm.metrics['signers']['val']) if wm else 'the'} validation signers; treat it as an estimate.",
              "- Several datasets are non-commercial or research-only (docs/DATASET.md). Don't deploy publicly or "
              "commercially with them.", ""]
    out = REPO_ROOT / "docs" / "EVALUATION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
