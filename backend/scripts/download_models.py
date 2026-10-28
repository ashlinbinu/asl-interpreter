"""Download the pinned MediaPipe .task models into $HANDSPEAK_MODELS_DIR/mediapipe (verifies SHA-256).

    python -m scripts.download_models
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mediapipe_models import MODELS, model_path  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for name, (url, digest) in MODELS.items():
        dst = model_path(name)
        if dst.exists() and sha256(dst) == digest:
            print(f"ok       {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        print(f"download {url}")
        urllib.request.urlretrieve(url, tmp)
        got = sha256(tmp)
        if got != digest:
            tmp.unlink()
            raise SystemExit(f"SHA-256 mismatch for {name}: {got} != {digest}")
        tmp.replace(dst)
        print(f"saved    {dst}")


if __name__ == "__main__":
    main()
