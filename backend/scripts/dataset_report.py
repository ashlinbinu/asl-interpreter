"""Clip/signer counts from the database -> <store>/reports/dataset_counts.json, and the Counts table of
docs/DATASET.md with --write-docs (between the <!-- counts:start --> / <!-- counts:end --> markers).

    python -m scripts.dataset_report --write-docs
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.db import ReferenceClip, Session, Sign, init_db  # noqa: E402
from app.registry import norm  # noqa: E402
from app.vocabulary import MIN_CLIPS, MIN_SIGNERS, WORDS  # noqa: E402

START, END = "<!-- counts:start -->", "<!-- counts:end -->"


def counts() -> dict:
    init_db()
    with Session() as s:
        rows = (s.query(Sign.category, Sign.label, ReferenceClip.source, ReferenceClip.signer_id, ReferenceClip.split)
                .join(ReferenceClip, ReferenceClip.sign_id == Sign.id).all())
    groups = defaultdict(lambda: {"signs": set(), "clips": 0, "signers": set(), "split": defaultdict(int)})
    per_sign = defaultdict(lambda: {"clips": 0, "signers": set()})
    for cat, label, src, signer, split in rows:
        g = groups[(cat or "unknown", src)]
        g["signs"].add(label)
        g["clips"] += 1
        g["signers"].add(f"{src}:{signer}")
        g["split"][split or "none"] += 1
        p = per_sign[(cat, norm(label))]
        p["clips"] += 1
        p["signers"].add(f"{src}:{signer}")
    table = [{"category": c, "source": src, "signs": len(g["signs"]), "clips": g["clips"],
              "signers": len(g["signers"]), "split": dict(g["split"])} for (c, src), g in sorted(groups.items())]
    words = {}
    for w in WORDS:
        p = per_sign.get(("word", norm(w)), {"clips": 0, "signers": set()})
        words[w] = {"clips": p["clips"], "signers": len(p["signers"]),
                    "ok": p["clips"] >= MIN_CLIPS and len(p["signers"]) >= MIN_SIGNERS}
    letters = {}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        p = per_sign.get(("alphabet", ch.lower()), {"clips": 0, "signers": set()})
        letters[ch] = {"clips": p["clips"], "signers": len(p["signers"])}
    return {"date": date.today().isoformat(), "groups": table, "words": words, "letters": letters}


def to_markdown(c: dict) -> str:
    lines = [f"Generated {c['date']} by `python -m scripts.dataset_report --write-docs`. Signers are counted per "
             "source (`source:signer`).", "",
             "| Category | Source | Signs | Clips | Signers | Clips by split |", "|---|---|---|---|---|---|"]
    if not c["groups"]:
        lines.append("| - | - | 0 | 0 | 0 | no clips ingested yet |")
    for g in c["groups"]:
        sp = ", ".join(f"{k} {v}" for k, v in sorted(g["split"].items()))
        lines.append(f"| {g['category']} | {g['source']} | {g['signs']} | {g['clips']} | {g['signers']} | {sp} |")
    ok = [w for w, v in c["words"].items() if v["ok"]]
    short = [f"{w} ({v['clips']}c/{v['signers']}s)" for w, v in c["words"].items() if not v["ok"]]
    lines += ["", f"Words with ≥{MIN_CLIPS} clips and ≥{MIN_SIGNERS} signers: **{len(ok)} of {len(c['words'])}**."]
    if short:
        lines.append(f"Below threshold: {', '.join(short)}.")
    lt = c["letters"]
    have = [ch for ch, v in lt.items() if v["clips"]]
    min_signers = min((v["signers"] for v in lt.values()), default=0)
    lines.append(f"Letters with clips: {len(have)} of 26; fewest signers for any letter: {min_signers}.")
    return "\n".join(lines)


def write_docs(md: str, path: Path):
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{md}\n{END}"
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-docs", action="store_true", help="update docs/DATASET.md")
    a = ap.parse_args()
    c = counts()
    out = get_settings().store_dir / "reports" / "dataset_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c, indent=2), encoding="utf-8")
    md = to_markdown(c)
    sys.stdout.reconfigure(errors="replace")  # Windows consoles (cp1252) can't print every character
    print(md)
    if a.write_docs:
        write_docs(md, REPO_ROOT / "docs" / "DATASET.md")
        print("Updated docs/DATASET.md")


if __name__ == "__main__":
    main()
