"""TEST ONLY: a local HTTP server with Range support, and a tiny fake ASL Citizen zip (no real data)."""
import csv
import http.server
import io
import re
import threading
import zipfile
from contextlib import contextmanager


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        data, srv = self.server.blob, self.server
        m = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range", ""))
        if m and srv.honor_range:
            start = int(m[1])
            end = min(int(m[2]) if m[2] else len(data) - 1, len(data) - 1)
            body = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if srv.truncate and len(body) > 1000:     # simulate a dropped connection mid-body
            srv.truncate -= 1
            body = body[:len(body) // 2]
            self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


@contextmanager
def serve(blob: bytes, honor_range=True, truncate=0):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.blob, srv.honor_range, srv.truncate = blob, honor_range, truncate
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/ASL_Citizen.zip"
    finally:
        srv.shutdown()
        srv.server_close()


# gloss -> [(user, split)]. Users P1-P10 train, P11-P12 val, P13-P16 test (disjoint, like the real splits).
def _clips(n, users, splits):
    return [(users[i % len(users)], splits[i % len(splits)]) for i in range(n)]


TRAIN = [f"P{i}" for i in range(1, 11)]
DEFAULT_GLOSSES = {
    "HELLO": [(u, "train") for u in TRAIN] * 2 + [("P11", "val"), ("P13", "test"), ("P14", "test")],
    "HELLO2": _clips(4, TRAIN[:3], ["train"]),
    "THANKYOU": [(u, "train") for u in TRAIN] + [("P11", "val"), ("P12", "val")] + _clips(4, ["P13", "P14"], ["test"]),
    "MOTHER": _clips(10, TRAIN[:5], ["train"]),
    "TOILET": [(u, "train") for u in TRAIN] + _clips(6, ["P15", "P16"], ["test"]),
    "A": _clips(12, TRAIN[:6], ["train"]),
    "I": _clips(3, TRAIN[:3], ["train"]),
}


def build_zip(glosses=None, video_bytes=3000, big_member=0) -> bytes:
    glosses = glosses or DEFAULT_GLOSSES
    buf = io.BytesIO()
    rows = {"train": [], "val": [], "test": []}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for gloss, items in glosses.items():
            for i, (user, split) in enumerate(items):
                fname = f"{user}-{gloss}-{i:03d}.mp4"
                rows[split].append((user, fname, gloss, f"code_{gloss.lower()}"))
                z.writestr(f"ASL_Citizen/videos/{fname}", bytes([i % 251]) * video_bytes)
        if big_member:
            z.writestr("ASL_Citizen/videos/zz-unused.mp4", b"\0" * big_member)
        for split, rs in rows.items():
            s = io.StringIO()
            w = csv.writer(s)
            w.writerow(["Participant ID", "Video file", "Gloss", "ASL-LEX Code"])
            w.writerows(rs)
            z.writestr(f"ASL_Citizen/splits/{split}.csv", s.getvalue())
    return buf.getvalue()
