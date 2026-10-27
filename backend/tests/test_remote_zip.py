"""Remote ZIP reading over HTTP ranges: transfers only what is read, never falls back to a full download."""
import io
import random
import zipfile

import pytest

from app.remote_zip import HttpRangeFile, RangeError
from tests.fake_citizen import build_zip, serve


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_reads_members_and_transfers_only_the_index():
    blob = build_zip(big_member=3_000_000)
    local = zipfile.ZipFile(io.BytesIO(blob))
    with serve(blob) as url:
        f = HttpRangeFile(url)
        with zipfile.ZipFile(f) as zf:
            assert zf.namelist() == local.namelist()
            assert zf.read("ASL_Citizen/splits/train.csv") == local.read("ASL_Citizen/splits/train.csv")
            name = next(n for n in zf.namelist() if n.endswith("-000.mp4"))
            assert zf.read(name) == local.read(name)
        assert f.size == len(blob)
        assert f.downloaded < len(blob) / 10


def test_random_seeks_match_the_bytes():
    blob = bytes(random.Random(1).randrange(256) for _ in range(300_000))
    with serve(blob) as url:
        f = HttpRangeFile(url, block=4096)
        rng = random.Random(2)
        for _ in range(40):
            pos, n = rng.randrange(len(blob)), rng.choice([1, 30, 4095, 4096, 9000, 70_000])
            f.seek(pos)
            assert f.read(n) == blob[pos:pos + n]
            assert f.tell() == min(pos + n, len(blob))
        f.seek(-10, io.SEEK_END)
        assert f.read() == blob[-10:]


def test_refuses_a_server_that_ignores_ranges():
    with serve(b"x" * 2_000_000, honor_range=False) as url, pytest.raises(RangeError, match="ignored the Range"):
        HttpRangeFile(url)


def test_transfer_budget_is_enforced():
    with serve(b"y" * 500_000) as url:
        f = HttpRangeFile(url, max_bytes=100_000)
        f.read(50_000)
        with pytest.raises(RangeError, match="budget"):
            f.read(100_000)
