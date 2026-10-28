"""Ingest the ASL Citizen clips for the word vocabulary: fetch only those videos (HTTP ranges), extract landmarks.

    python -m scripts.datasets.ingest_asl_citizen --dry-run        # plan + bytes to transfer; downloads nothing
    python -m scripts.datasets.ingest_asl_citizen --max-gb 2       # fetch + extract (resumable, re-run safely)
    python -m scripts.datasets.ingest_asl_citizen --word bathroom=TOILET --max-per-gloss 20 --workers 3

Each MediaPipe worker holds ~740 MB and many threads: beyond 3-4 workers a 16 GB laptop slows down, not up.

Videos land in <store>/raw/asl_citizen/videos/, landmarks in <store>/landmarks/asl_citizen/<GLOSS>/. Every clip keeps
its official split (train/val/test are disjoint signer sets), signer id, license and attribution.
ASL Citizen's license allows non-commercial, non-revenue research use only; accept Microsoft's terms first.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.datasets.asl_citizen import ATTRIBUTION, LICENSE, SOURCE, SOURCE_URL, URL, Archive  # noqa: E402
from scripts.datasets.common import add_clip, existing_keys, setup, upsert_word  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Session  # noqa: E402
from app.features import MIN_HAND_FRAMES  # noqa: E402
from app.vocabulary import WORDS, match_words, summarize  # noqa: E402

SPLITS = ("train", "val", "test")


def resolve(rows, words=WORDS, overrides=None, include_low=False):
    """-> ({word: gloss}, {word: reason skipped}). Synonym glosses are only used when given as an override."""
    overrides = overrides or {}
    stats = summarize(rows)
    mapping, skipped = {}, {}
    for w, v in match_words(stats, words).items():
        if w in overrides:
            if overrides[w] not in stats:
                raise SystemExit(f"--word {w}={overrides[w]}: no such gloss in the archive")
            mapping[w] = overrides[w]
        elif v["status"] == "ok" or (include_low and v["status"] == "low"):
            mapping[w] = v["chosen"]
        else:
            skipped[w] = v["status"] + (f" (candidate {v['chosen']})" if v["chosen"] else "")
    return mapping, skipped


def select(rows, mapping, splits=SPLITS, max_per_gloss=0):
    """[(word, row)]; with max_per_gloss, clips are taken round-robin over signers (deterministic)."""
    by_gloss = defaultdict(list)
    for r in rows:
        if r.split in splits:
            by_gloss[r.gloss].append(r)
    out = []
    for word, gloss in mapping.items():
        rs = sorted(by_gloss.get(gloss, []), key=lambda r: (r.user, r.video))
        if max_per_gloss and len(rs) > max_per_gloss:
            per_user = defaultdict(list)
            for r in rs:
                per_user[r.user].append(r)
            queues, picked = [per_user[u] for u in sorted(per_user)], []
            while len(picked) < max_per_gloss:
                for q in queues:
                    if q and len(picked) < max_per_gloss:
                        picked.append(q.pop(0))
            rs = picked
        out += [(word, r) for r in rs]
    return out


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "x"


def _extract(job):
    """Worker: (video path, max side) -> (frames, fps, error). Top-level so it pickles on Windows."""
    path, max_side = job
    try:
        from app.extract import video_frames
        frames, fps = video_frames(Path(path), max_side=max_side)
        return frames, fps, None
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the run
        return None, None, str(e)


def ingest(src=URL, words=WORDS, overrides=None, splits=SPLITS, max_per_gloss=0, max_gb=3.0, workers=0,
           max_side=720, include_low=False, dry_run=False, extract=None, log=print) -> dict:
    store = get_settings().store_dir
    raw_dir = store / "raw" / SOURCE / "videos"
    setup()
    t0 = time.time()
    with Archive(src) as arc:
        rows = [r for r in arc.rows() if arc.has_video(r.video)]
        mapping, skipped = resolve(rows, words, overrides, include_low)
        todo = select(rows, mapping, splits, max_per_gloss)
        rel = {id(r): f"landmarks/{SOURCE}/{_safe(mapping[w])}/{_safe(Path(r.video).stem)}.npz" for w, r in todo}
        done = existing_keys(SOURCE)
        pending = [(w, r) for w, r in todo if rel[id(r)] not in done]
        need = [(w, r) for w, r in pending if not (raw_dir / r.video).exists()]
        plan = {"words": len(mapping), "skipped_words": skipped, "clips_selected": len(todo),
                "clips_already_ingested": len(todo) - len(pending), "videos_to_fetch": len(need),
                "bytes_to_fetch": sum(r.size for _, r in need), "store": str(store)}
        log(f"{plan['words']} words -> {plan['clips_selected']} clips ({plan['clips_already_ingested']} already "
            f"ingested); fetch {plan['videos_to_fetch']} videos = {plan['bytes_to_fetch'] / 1e6:,.1f} MB "
            f"into {raw_dir}")
        for w, why in skipped.items():
            log(f"  skipped {w}: {why}")
        if dry_run:
            return plan
        if plan["bytes_to_fetch"] > max_gb * 1e9:
            raise SystemExit(f"Planned transfer {plan['bytes_to_fetch'] / 1e9:.2f} GB exceeds --max-gb {max_gb}")
        if arc.http:  # hard cap: the plan plus a margin for zip headers
            arc.http.max_bytes = arc.downloaded + plan["bytes_to_fetch"] + (64 << 20)
        for i, (_, r) in enumerate(need, 1):
            arc.fetch_video(r.video, raw_dir / r.video)
            if i % 25 == 0 or i == len(need):
                log(f"  fetched {i}/{len(need)} ({arc.downloaded / 1e6:,.0f} MB) [{time.time() - t0:.0f}s]")
        plan["bytes_transferred"] = arc.downloaded

    jobs = [(str(raw_dir / r.video), max_side) for _, r in pending]
    if extract is not None:
        results = map(extract, jobs)
    elif workers > 1:
        pool = ProcessPoolExecutor(workers)
        results = pool.map(_extract, jobs, chunksize=1)
    else:
        results = map(_extract, jobs)
    per_word = defaultdict(lambda: {"clips": 0, "signers": set(), "split": defaultdict(int), "no_hands": 0,
                                    "errors": 0})
    try:
        with Session() as s:
            signs = {w: upsert_word(s, w, SOURCE) for w in mapping}
            s.commit()
            for k, ((w, r), (frames, fps, err)) in enumerate(zip(pending, results), 1):
                pw = per_word[w]
                if err or frames is None:
                    pw["errors"] += 1
                    log(f"  {r.video}: {err}")
                    continue
                if sum(1 for f in frames if f["hands"]) < MIN_HAND_FRAMES:
                    pw["no_hands"] += 1
                    continue
                add_clip(s, signs[w], frames, source=SOURCE, signer_id=r.user, rel_path=rel[id(r)],
                         license=LICENSE, attribution=ATTRIBUTION, source_url=SOURCE_URL,
                         video_path=(raw_dir / r.video).relative_to(store).as_posix(), fps=fps, split=r.split,
                         extra_quality={"gloss": r.gloss})
                pw["clips"] += 1
                pw["signers"].add(r.user)
                pw["split"][r.split] += 1
                if k % 20 == 0:
                    s.commit()
                    log(f"  extracted {k}/{len(pending)} [{time.time() - t0:.0f}s]")
            s.commit()
    finally:
        if extract is None and workers > 1:
            pool.shutdown()
    plan["ingested"] = {w: {"gloss": mapping[w], "clips": v["clips"], "signers": len(v["signers"]),
                            "split": dict(v["split"]), "no_hands": v["no_hands"], "errors": v["errors"]}
                        for w, v in per_word.items()}
    plan["seconds"] = round(time.time() - t0, 1)
    out = store / "reports" / "ingest_asl_citizen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log(f"Done in {plan['seconds']}s. Report: {out}. Next: python -m scripts.build_reference_videos, "
        "python -m scripts.dataset_report --write-docs, then restart the API.")
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=URL, help="archive URL (default: official), local .zip, or extracted folder")
    ap.add_argument("--word", action="append", default=[], metavar="WORD=GLOSS", help="force a gloss for a word")
    ap.add_argument("--only", default=None, help="comma-separated subset of the vocabulary")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--max-per-gloss", type=int, default=0, help="0 = all clips")
    ap.add_argument("--max-gb", type=float, default=3.0, help="refuse to start if the transfer would be larger")
    ap.add_argument("--workers", type=int, default=1, help="parallel landmark extraction processes")
    ap.add_argument("--max-side", type=int, default=720, help="downscale frames to this many pixels before MediaPipe")
    ap.add_argument("--include-low", action="store_true", help="also ingest words below the clip/signer thresholds")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    overrides = dict(w.split("=", 1) for w in a.word)
    words = [w.strip() for w in a.only.split(",")] if a.only else WORDS
    ingest(a.src, words, overrides, tuple(a.splits.split(",")), a.max_per_gloss, a.max_gb, a.workers, a.max_side,
           a.include_low, a.dry_run)


if __name__ == "__main__":
    main()
