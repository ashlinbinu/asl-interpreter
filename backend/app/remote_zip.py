"""Read members of a remote ZIP (e.g. the 46 GB ASL Citizen archive) with HTTP range requests.

Only the central directory and the members you open are transferred. A server that ignores the Range header is
detected on the first request and the connection is closed before the body is read, so this can never turn into a
full download by accident. `max_bytes` caps the total transfer.

    with zipfile.ZipFile(HttpRangeFile(url, max_bytes=64 << 20)) as zf:
        data = zf.read("ASL_Citizen/splits/train.csv")
"""
import io
import time
import urllib.error
import urllib.request


class RangeError(OSError):
    """The server can't serve byte ranges, or the transfer budget would be exceeded."""


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, with a small read-ahead block for header reads."""

    def __init__(self, url: str, *, block: int = 64 << 10, max_bytes: int | None = None, timeout: float = 60,
                 retries: int = 4, user_agent: str = "handspeak-dataset-ingest"):
        super().__init__()
        self.url, self.block, self.max_bytes = url, block, max_bytes
        self.timeout, self.retries, self.user_agent = timeout, retries, user_agent
        self.pos = 0
        self.downloaded = 0          # body bytes received
        self.requests = 0
        self._buf_start, self._buf = 0, b""
        self._opener = urllib.request.build_opener()   # proxy settings read now, not cached process-wide
        self.size = self._probe()

    def _open(self, start: int, end: int):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                                        "User-Agent": self.user_agent})
        r = self._opener.open(req, timeout=self.timeout)
        if r.status != 206:
            r.close()
            raise RangeError(f"{self.url}: server ignored the Range header (HTTP {r.status}); refusing to download")
        return r

    def _probe(self) -> int:
        with self._open(0, 0) as r:
            total = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
            self.downloaded += len(r.read())
            self.requests += 1
        if not total.isdigit():
            raise RangeError(f"{self.url}: response has no total size in Content-Range")
        return int(total)

    def _fetch(self, start: int, n: int) -> bytes:
        end = min(start + n, self.size) - 1
        if end < start:
            return b""
        want = end - start + 1
        if self.max_bytes is not None and self.downloaded + want > self.max_bytes:
            raise RangeError(f"transfer budget exceeded ({self.downloaded + want:,} > {self.max_bytes:,} bytes)")
        for attempt in range(self.retries + 1):
            try:
                with self._open(start, end) as r:
                    data = r.read()
                if len(data) != want:
                    raise OSError(f"short read: {len(data)} of {want} bytes")
                self.downloaded += len(data)
                self.requests += 1
                return data
            except RangeError:
                raise
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 or attempt == self.retries:
                    raise
            except OSError:
                if attempt == self.retries:
                    raise
            time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    # io.RawIOBase interface ------------------------------------------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        left = max(self.size - self.pos, 0)
        n = left if n is None or n < 0 else min(n, left)
        if n == 0:
            return b""
        out = b""
        off = self.pos - self._buf_start
        if 0 <= off < len(self._buf):
            out = self._buf[off:off + n]
        rest = n - len(out)
        if rest > 0:
            p = self.pos + len(out)
            if rest >= self.block:
                out += self._fetch(p, rest)
            else:
                self._buf_start, self._buf = p, self._fetch(p, self.block)
                out += self._buf[:rest]
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
