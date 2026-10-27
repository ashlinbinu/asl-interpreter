"""DEV ONLY: (re)write the old procedurally generated templates into signs.frames (frames_source='synthetic').

They are not real signing. They are used only when HANDSPEAK_DEV_SYNTHETIC=1, for UI work before real
reference clips exist.

    python -m scripts.seed_dev_synthetic
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.datasets.common import find_sign, setup  # noqa: E402

from app.db import Session, Sign  # noqa: E402
from app.legacy.synthetic import get_all_canonical_signs  # noqa: E402


def main():
    setup()
    n = 0
    with Session() as s:
        for item in get_all_canonical_signs():
            cat = "alphabet" if item["source"] == "alphabet" else "word"
            row = find_sign(s, item["label"], cat)
            if row is None:
                row = Sign(label=item["label"], gloss=item["label"].upper().replace(" ", "-"), category=cat,
                           kind="dynamic", source=item["source"])
                s.add(row)
            row.frames, row.frames_source = item["frames"], "synthetic"
            n += 1
        s.commit()
    print(f"{n} synthetic templates written (dev fallback only; set HANDSPEAK_DEV_SYNTHETIC=1 to use them).")


if __name__ == "__main__":
    main()
