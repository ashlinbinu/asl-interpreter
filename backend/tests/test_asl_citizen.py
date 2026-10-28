"""ASL Citizen pipeline on a fake archive: inspection, gloss matching, selective ingest, reference videos, counts."""
import numpy as np
import pytest

from app.vocabulary import gloss_keys, letter_of, match_words, summarize
from scripts.datasets.asl_citizen import Archive
from scripts.datasets.inspect_asl_citizen import inspect, to_markdown
from tests import synth
from tests.fake_citizen import build_zip, serve


@pytest.fixture(scope="module")
def zip_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("citizen") / "ASL_Citizen.zip"
    p.write_bytes(build_zip())
    return p


def test_gloss_keys_and_letters():
    assert gloss_keys("THANK-YOU") == {"THANKYOU"}
    assert "HELLO" in gloss_keys("HELLO2")
    assert {"DONE", "FINISH"} <= gloss_keys("DONE (FINISH)")
    assert {"WHO", "WHOM"} <= gloss_keys("WHO/WHOM")
    assert [letter_of(g) for g in ("A", "B2", "LETTER-C", "FS-D", "#E", "AB", "HELLO")] == \
        ["A", "B", "C", "D", "E", None, None]


def test_archive_reads_splits_by_signer(zip_path):
    with Archive(str(zip_path)) as arc:
        rows = arc.rows()
    assert {r.split for r in rows} == {"train", "val", "test"}
    users = {s: {r.user for r in rows if r.split == s} for s in ("train", "val", "test")}
    assert not (users["train"] & users["val"]) and not (users["train"] & users["test"])
    assert all(r.size == 3000 for r in rows)


def test_inspect_reports_letters_vocab_and_bytes(zip_path):
    rep = inspect(str(zip_path), words=["hello", "thank-you", "mother", "bathroom", "goodbye"])
    assert rep["totals"]["glosses"] == 7
    assert [s["gloss"] for s in rep["letters"]["A"]] == ["A"]
    assert rep["letters"]["A"][0]["signers"] == 6
    assert [s["gloss"] for s in rep["letters"]["I"]] == ["I"]
    assert rep["letters"]["Z"] == []
    v = rep["vocabulary"]
    assert (v["hello"]["status"], v["hello"]["chosen"]) == ("ok", "HELLO")
    assert [s["gloss"] for s in v["hello"]["exact"]] == ["HELLO", "HELLO2"]
    assert (v["thank-you"]["status"], v["thank-you"]["chosen"]) == ("ok", "THANKYOU")
    assert v["mother"]["status"] == "low"
    assert (v["bathroom"]["status"], v["bathroom"]["chosen"]) == ("alias", "TOILET")
    assert v["goodbye"]["status"] == "missing"
    est = rep["ingest_estimate"]
    assert est["clips"] == 23 + 16 + 10 + 16 and est["bytes"] == est["clips"] * 3000
    md = to_markdown(rep)
    assert "| hello | ok | `HELLO` |" in md and "2 of 26 letters" in md


def test_inspect_over_http_reads_only_the_index(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    blob = build_zip(big_member=4_000_000)
    with serve(blob) as url:
        rep = inspect(url, words=["hello"], max_bytes=2_000_000)
    assert rep["totals"]["clips"] == 84
    assert rep["transferred_bytes"] < 300_000 < len(blob)


def test_select_spreads_over_signers(zip_path):
    from scripts.datasets.ingest_asl_citizen import select
    with Archive(str(zip_path)) as arc:
        rows = arc.rows()
    picked = select(rows, {"hello": "HELLO"}, ("train",), max_per_gloss=10)
    assert len(picked) == 10 and len({r.user for _, r in picked}) == 10


def _fake_extract(job):
    """Landmarks for a fake clip: a moving hand, except videos named *-007.mp4 have no hand."""
    path, _ = job
    if path.endswith("-007.mp4"):
        return [{"t": i * 33.3, "w": 640, "h": 480, "hands": [], "pose": None} for i in range(10)], 30.0, None
    rng = np.random.default_rng(len(path))
    frames = []
    for i in range(12):
        lm = synth.hand((0.4 + 0.01 * i, 0.5), rng=rng, noise=0.002)
        frames.append({"t": i * 33.3, "w": 640, "h": 480, "pose": synth.pose(),
                       "hands": [{"handedness": "Left", "score": 0.9, "lm": lm.tolist()}]})
    return frames, 30.0, None


def test_ingest_build_and_report(client, zip_path, tmp_root, tmp_path):
    from app.config import get_settings
    from app.db import ReferenceClip, ReferenceVideo, Session, Sign
    from scripts.build_reference_videos import build
    from scripts.dataset_report import START, counts, to_markdown, write_docs
    from scripts.datasets.ingest_asl_citizen import ingest

    words = ["hello", "thank-you", "mother", "bathroom"]
    plan = ingest(str(zip_path), words, overrides={"bathroom": "TOILET"}, dry_run=True, log=lambda *_: None)
    assert plan["words"] == 3 and "mother" in plan["skipped_words"]
    assert plan["videos_to_fetch"] == 23 + 16 + 16 and plan["bytes_to_fetch"] == 55 * 3000

    rep = ingest(str(zip_path), words, overrides={"bathroom": "TOILET"}, extract=_fake_extract, log=lambda *_: None)
    no_hands = sum(v["no_hands"] for v in rep["ingested"].values())
    assert no_hands >= 1
    with Session() as s:
        clips = s.query(ReferenceClip).filter(ReferenceClip.source == "asl_citizen").all()
        assert len(clips) == 55 - no_hands
        assert all(c.license.startswith("ASL Citizen") and c.split in ("train", "val", "test") for c in clips)
        hello = next(r for r in s.query(Sign) if r.label == "hello")
        assert (hello.category, hello.kind, hello.gloss) == ("word", "dynamic", "HELLO")
        assert {c.signer_id for c in clips if c.split == "test"} <= {"P13", "P14", "P15", "P16"}
    store = get_settings().store_dir
    assert all((store / c.video_path).is_file() and (store / c.landmarks_path).is_file() for c in clips)

    again = ingest(str(zip_path), words, overrides={"bathroom": "TOILET"}, extract=_fake_extract,
                   log=lambda *_: None)
    assert again["videos_to_fetch"] == 0 and again["clips_already_ingested"] == 55 - no_hands

    out = build(category="word", log=lambda *_: None)
    assert len(out["published"]) == 3 and all(p["split"] == "train" for p in out["published"])
    with Session() as s:
        for v in s.query(ReferenceVideo):
            assert (get_settings().media_dir / v.path).is_file() and v.license.startswith("ASL Citizen")
            assert s.query(ReferenceClip).filter(ReferenceClip.sign_id == v.sign_id,
                                                 ReferenceClip.is_canonical.is_(True)).count() == 1
    assert build(category="word", log=lambda *_: None)["kept"] == 3

    c = counts()
    assert any(g["source"] == "asl_citizen" and g["category"] == "word" for g in c["groups"])
    assert c["words"]["hello"]["ok"] and not c["words"]["mother"]["ok"]
    doc = tmp_path / "DATASET.md"
    doc.write_text("# Datasets\n\n## Counts\n\n" + START + "\nold\n<!-- counts:end -->\n\nTail\n", encoding="utf-8")
    write_docs(to_markdown(c), doc)
    text = doc.read_text(encoding="utf-8")
    assert "\nold\n" not in text and "| word | asl_citizen |" in text and text.endswith("Tail\n")


def test_match_words_prefers_most_clips():
    class R:
        def __init__(self, gloss, user):
            self.gloss, self.user, self.split, self.size = gloss, user, "train", 1
    stats = summarize([R("SIGN1", f"u{i}") for i in range(3)] + [R("SIGN2", f"u{i}") for i in range(20)])
    assert match_words(stats, ["sign"])["sign"]["chosen"] == "SIGN2"
