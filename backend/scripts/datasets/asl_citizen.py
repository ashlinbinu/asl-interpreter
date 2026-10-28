"""ASL Citizen archive access (Microsoft Research, 2023): the remote zip over HTTP ranges, a local zip, or an extracted
folder. Layout (from microsoft/ASL-citizen-code): ASL_Citizen/videos/<file>.mp4 and ASL_Citizen/splits/{train,val,
test}.csv with columns Participant ID, Video file, Gloss, ASL-LEX Code. Splits are by signer.
"""
import csv
import io
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.remote_zip import HttpRangeFile  # noqa: E402

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
SOURCE = "asl_citizen"
LICENSE = "ASL Citizen dataset license (Microsoft Research): non-commercial, non-revenue research use only"
ATTRIBUTION = "ASL Citizen (Desai et al., NeurIPS 2023 Datasets & Benchmarks), Microsoft Research"
SOURCE_URL = "https://www.microsoft.com/en-us/research/project/asl-citizen/"
PUBLISHED = {"glosses": 2731, "clips": 83399, "signers": 52}

_SPLIT_CSV = re.compile(r"(?:^|/)(train|val|test)\.csv$", re.I)


@dataclass
class Row:
    user: str
    video: str          # file name as given in the CSV
    gloss: str
    code: str | None
    split: str
    size: int = 0       # bytes as stored in the archive (what a fetch transfers)


def _columns(header: list[str]) -> dict[str, int]:
    """Map our fields to CSV column indexes by header name, falling back to the documented order."""
    cols = {"user": 0, "video": 1, "gloss": 2, "code": 3}
    for i, h in enumerate(h.strip().lower() for h in header):
        if "participant" in h or h in ("user", "user id", "signer"):
            cols["user"] = i
        elif "video" in h or h in ("file", "filename"):
            cols["video"] = i
        elif h == "gloss":
            cols["gloss"] = i
        elif "asl-lex" in h or h == "code":
            cols["code"] = i
    return cols


class Archive:
    """Uniform read access to the dataset. `downloaded` counts bytes fetched over HTTP."""

    def __init__(self, src: str, max_bytes: int | None = None):
        self.src, self.http, self.zf, self.root = str(src), None, None, None
        if self.src.startswith(("http://", "https://")):
            self.http = HttpRangeFile(self.src, max_bytes=max_bytes)
            self.zf = zipfile.ZipFile(self.http)
        elif Path(self.src).is_dir():
            self.root = Path(self.src)
        else:
            self.zf = zipfile.ZipFile(self.src)
        self._videos = {}
        self._csvs = {}
        if self.zf:
            for info in self.zf.infolist():
                p = PurePosixPath(info.filename)
                if info.is_dir():
                    continue
                if p.parent.name == "videos":
                    self._videos[p.name] = info
                elif m := _SPLIT_CSV.search(info.filename):
                    self._csvs[m.group(1).lower()] = info.filename
        else:
            for f in self.root.rglob("*"):
                if f.is_file() and f.parent.name == "videos":
                    self._videos[f.name] = f
                elif f.is_file() and (m := _SPLIT_CSV.search(f.as_posix())):
                    self._csvs[m.group(1).lower()] = f

    @property
    def downloaded(self) -> int:
        return self.http.downloaded if self.http else 0

    @property
    def size(self) -> int | None:
        if self.http:
            return self.http.size
        return None if self.root else Path(self.src).stat().st_size

    def close(self):
        if self.zf:
            self.zf.close()
        if self.http:
            self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _read(self, ref) -> bytes:
        return self.zf.read(ref) if self.zf else Path(ref).read_bytes()

    def video_size(self, name: str) -> int:
        ref = self._videos.get(name)
        if ref is None:
            return 0
        return ref.compress_size if self.zf else ref.stat().st_size

    def has_video(self, name: str) -> bool:
        return name in self._videos

    def rows(self) -> list[Row]:
        if not self._csvs:
            raise FileNotFoundError(f"No splits/train|val|test.csv found in {self.src}")
        out = []
        for split in sorted(self._csvs):
            text = self._read(self._csvs[split]).decode("utf-8-sig")
            reader = csv.reader(io.StringIO(text))
            header = next(reader, None) or []
            c = _columns(header)
            for r in reader:
                if len(r) <= max(c["user"], c["video"], c["gloss"]):
                    continue
                name = PurePosixPath(r[c["video"]].strip().replace("\\", "/")).name
                out.append(Row(user=r[c["user"]].strip(), video=name, gloss=r[c["gloss"]].strip(),
                               code=r[c["code"]].strip() if len(r) > c["code"] else None, split=split,
                               size=self.video_size(name)))
        return out

    def fetch_video(self, name: str, dest: Path) -> Path:
        """Copy one clip to `dest` (atomic: written to .part first). Only this member is transferred."""
        ref = self._videos[name]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        if self.zf:
            with self.zf.open(ref) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
        else:
            shutil.copyfile(ref, tmp)
        tmp.replace(dest)
        return dest
