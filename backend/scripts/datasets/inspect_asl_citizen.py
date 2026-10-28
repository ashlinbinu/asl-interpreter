"""Report what ASL Citizen contains for our vocabulary and the alphabet, without downloading any video.

Reads only the zip's central directory and the three split CSVs (HTTP range requests, capped by --max-mb), then
reports per-gloss clip and signer counts by split, letter glosses (A-Z), the 50-word vocabulary coverage, and how
many bytes ingesting the chosen glosses would transfer.

    python -m scripts.datasets.inspect_asl_citizen                    # remote zip (default URL)
    python -m scripts.datasets.inspect_asl_citizen --src D:\\data\\ASL_Citizen.zip
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.datasets.asl_citizen import PUBLISHED, URL, Archive  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.vocabulary import MIN_CLIPS, MIN_SIGNERS, WORDS, match_letters, match_words, summarize  # noqa: E402


def inspect(src: str, words=WORDS, max_bytes: int | None = 64 << 20) -> dict:
    with Archive(src, max_bytes=max_bytes) as arc:
        rows = arc.rows()
        stats = summarize(rows)
        letters = match_letters(stats)
        vocab = match_words(stats, words)
        splits = {}
        for r in rows:
            d = splits.setdefault(r.split, {"clips": 0, "signers": set()})
            d["clips"] += 1
            d["signers"].add(r.user)
        missing_videos = sum(1 for r in rows if not arc.has_video(r.video))
        chosen = [v["chosen"] for v in vocab.values() if v["chosen"]]
        return {
            "source": src,
            "inspected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "archive_bytes": arc.size,
            "transferred_bytes": arc.downloaded,
            "totals": {"glosses": len(stats), "clips": len(rows), "signers": len({r.user for r in rows}),
                       "published": PUBLISHED, "csv_rows_without_video": missing_videos,
                       "splits": {k: {"clips": v["clips"], "signers": len(v["signers"])}
                                  for k, v in sorted(splits.items())}},
            "letters": {ch: [stats[g] for g in gs] for ch, gs in letters.items()},
            "vocabulary": {w: {**v, "stats": stats.get(v["chosen"]),
                               "exact": [stats[g] for g in v["exact"]],
                               "aliases": [stats[g] for g in v["aliases"]]} for w, v in vocab.items()},
            "ingest_estimate": {"glosses": len(chosen), "clips": sum(stats[g]["clips"] for g in chosen),
                                "bytes": sum(stats[g]["bytes"] for g in chosen),
                                "train_clips": sum(stats[g]["split"].get("train", {}).get("clips", 0)
                                                   for g in chosen)},
            "thresholds": {"min_clips": MIN_CLIPS, "min_signers": MIN_SIGNERS},
        }


def _split_cell(s):
    return " / ".join(f"{s['split'].get(k, {}).get('clips', 0)}c {s['split'].get(k, {}).get('signers', 0)}s"
                      for k in ("train", "val", "test"))


def to_markdown(rep: dict) -> str:
    t = rep["totals"]
    mb = lambda b: f"{(b or 0) / 1e6:,.1f} MB"  # noqa: E731
    lines = [f"# ASL Citizen inspection ({rep['inspected_at']})", "",
             f"Source: `{rep['source']}`. Archive {mb(rep['archive_bytes'])}; this inspection transferred "
             f"{mb(rep['transferred_bytes'])} (index + split CSVs only, no videos).", "",
             f"Totals: **{t['glosses']} glosses, {t['clips']} clips, {t['signers']} signers** "
             f"(published: {t['published']['glosses']} / {t['published']['clips']} / {t['published']['signers']}). "
             f"CSV rows without a video in the archive: {t['csv_rows_without_video']}.", "",
             "| Split | Clips | Signers |", "|---|---|---|"]
    lines += [f"| {k} | {v['clips']} | {v['signers']} |" for k, v in t["splits"].items()]
    found = sum(1 for gs in rep["letters"].values() if gs)
    lines += ["", "## Letters (A-Z)", "",
              f"{found} of 26 letters have a candidate gloss. A one-letter gloss can be a word sign (e.g. the "
              "pronoun I), so check a clip before using it.", "",
              "| Letter | Glosses | Clips | Signers | train / val / test |", "|---|---|---|---|---|"]
    for ch, gs in rep["letters"].items():
        if not gs:
            lines.append(f"| {ch} | none | 0 | 0 | |")
        for s in gs:
            lines.append(f"| {ch} | `{s['gloss']}` | {s['clips']} | {s['signers']} | {_split_cell(s)} |")
    th = rep["thresholds"]
    lines += ["", "## Vocabulary", "",
              f"Status: `ok` = exact gloss with ≥{th['min_clips']} clips and ≥{th['min_signers']} signers; "
              "`alias` = only a synonym gloss exists (verify it is the same sign); `low` = below the thresholds; "
              "`missing` = no gloss.", "",
              "| Word | Status | Chosen gloss | Clips | Signers | train / val / test | Other candidates |",
              "|---|---|---|---|---|---|---|"]
    for w, v in rep["vocabulary"].items():
        s = v["stats"]
        others = [o["gloss"] + f" ({o['clips']})" for o in v["exact"] + v["aliases"] if o["gloss"] != v["chosen"]]
        lines.append(f"| {w} | {v['status']} | {('`' + v['chosen'] + '`') if v['chosen'] else '-'} | "
                     f"{s['clips'] if s else 0} | {s['signers'] if s else 0} | {_split_cell(s) if s else ''} | "
                     f"{', '.join(others)} |")
    counts = {k: sum(1 for v in rep["vocabulary"].values() if v["status"] == k)
              for k in ("ok", "alias", "low", "missing")}
    e = rep["ingest_estimate"]
    lines += ["", f"Coverage: {counts['ok']} ok, {counts['alias']} alias, {counts['low']} low, "
                  f"{counts['missing']} missing (of {len(rep['vocabulary'])}).", "",
              f"Ingesting the chosen glosses: {e['glosses']} glosses, {e['clips']} clips "
              f"({e['train_clips']} train), **{mb(e['bytes'])}** to transfer.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=URL, help="archive URL, local .zip, or extracted folder")
    ap.add_argument("--max-mb", type=float, default=64, help="abort if the inspection would transfer more than this")
    ap.add_argument("--out", type=Path, default=None, help="report folder (default <store>/reports)")
    a = ap.parse_args()
    rep = inspect(a.src, max_bytes=int(a.max_mb * 1e6))
    out = a.out or get_settings().store_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "asl_citizen_inspect.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = to_markdown(rep)
    (out / "asl_citizen_inspect.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")  # Windows consoles (cp1252) can't print every character
    print(md)
    print(f"Reports: {out / 'asl_citizen_inspect.md'} and .json")


if __name__ == "__main__":
    main()
