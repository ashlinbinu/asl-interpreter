"""Storage backends (local disk, S3-compatible bucket) and publishing the public part of a store to a hosted one."""
import io

import pytest

from app.storage import LocalStorage, S3Storage, media_key


class FakeS3:
    """In-memory stand-in for the boto3 S3 client calls S3Storage makes."""

    def __init__(self):
        self.objects = {}

    @staticmethod
    def _error(code):
        e = Exception(code)
        e.response = {"Error": {"Code": code}}
        return e

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._error("404")
        return {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._error("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = (Body, ContentType)

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://signed.example/{Params['Bucket']}/{Params['Key']}?op={op}&ttl={ExpiresIn}"


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path):
    return LocalStorage(tmp_path / "store") if request.param == "local" else S3Storage("bucket", client=FakeS3())


def test_round_trip(store):
    key = "landmarks/user/abc/hello_1.npz"
    assert not store.exists(key)
    store.write(key, b"xyz")
    assert store.exists(key) and store.read(key) == b"xyz"
    assert store.delete(key) is True
    assert store.delete(key) is False and store.delete(None) is False
    with pytest.raises(FileNotFoundError):
        store.read(key)


@pytest.mark.parametrize("key", ["../outside.txt", "media/../../outside.txt", ""])
def test_keys_cannot_leave_the_store(store, key):
    assert not store.exists(key) and store.delete(key) is False
    with pytest.raises(FileNotFoundError):
        store.read(key)
    with pytest.raises(ValueError):
        store.write(key, b"x")


def test_clips_round_trip_through_storage(store):
    from app.clipstore import read_clip, write_clip
    frames = [{"t": 0.0, "w": 640, "h": 480, "pose": None,
               "hands": [{"handedness": "Left", "score": 0.9, "lm": [[0.1, 0.2, 0.0]] * 21}]}]
    write_clip(store, "landmarks/user/abc/a.npz", frames, {"label": "a"})
    got, meta = read_clip(store, "landmarks/user/abc/a.npz")
    assert meta == {"label": "a"} and got[0]["hands"][0]["lm"][0] == pytest.approx([0.1, 0.2, 0.0])


def test_s3_content_types_and_signed_media_urls():
    fake = FakeS3()
    s3 = S3Storage("bucket", client=fake)
    s3.write("media/videos/3/user_1.webm", b"v")
    s3.write("landmarks/x.npz", b"n")
    assert fake.objects["media/videos/3/user_1.webm"][1] == "video/webm"
    assert fake.objects["landmarks/x.npz"][1] == "application/octet-stream"
    assert s3.url(media_key("videos/3/user_1.webm")).startswith(
        "https://signed.example/bucket/media/videos/3/user_1.webm?op=get_object")
    with pytest.raises(ValueError):
        s3.url("media/../landmarks/x.npz")
    with pytest.raises(ValueError, match="HANDSPEAK_S3_BUCKET"):
        S3Storage("")


@pytest.fixture()
def two_dbs(tmp_path):
    """A source and a target database at the current schema; the target already has the letter A, as a hosted app
    that has started once does."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db import Sign, migrate
    engines, sessions = [], []
    for name in ("src", "dst"):
        url = f"sqlite:///{(tmp_path / f'{name}.db').as_posix()}"
        migrate(url)
        engines.append(create_engine(url))
        sessions.append(sessionmaker(bind=engines[-1])())
    sessions[1].add(Sign(label="A", category="alphabet", kind="static", source="alphabet"))
    sessions[1].commit()
    yield sessions
    for s, e in zip(sessions, engines):
        s.close()
        e.dispose()


def test_publish_copies_only_public_media(two_dbs, tmp_path):
    from app.db import Attempt, ReferenceClip, ReferenceImage, ReferenceVideo, Sign
    from scripts.publish_store import publish
    src, dst = two_dbs
    src_store, dst_store = LocalStorage(tmp_path / "src-store"), S3Storage("bucket", client=FakeS3())
    a = Sign(label="A", category="alphabet", kind="static", source="alphabet")
    hello = Sign(label="hello", category="word", kind="dynamic", gloss="HELLO", source="asl_citizen")
    mine = Sign(label="my sign", category="user", kind="dynamic", source="user")
    src.add_all([a, hello, mine])
    src.flush()

    def clip(sign, source, key, canonical=False):
        src_store.write(key, b"npz " + key.encode())
        c = ReferenceClip(sign_id=sign.id, source=source, signer_id="s1", landmarks_path=key, license=source,
                          video_path=f"raw/{source}.mp4", is_canonical=canonical)
        src.add(c)
        src.flush()
        return c

    def media(model, sign, c, path):
        src_store.write(media_key(path), b"media " + path.encode())
        src.add(model(sign_id=sign.id, clip_id=c.id, path=path, license=c.license))

    commons = clip(a, "wikimedia_commons", "landmarks/wikimedia_commons/A.npz", canonical=True)
    photo = clip(a, "kaggle_asl_alphabet", "landmarks/kaggle_asl_alphabet/A1.npz")
    clip(a, "kaggle_asl_alphabet", "landmarks/kaggle_asl_alphabet/A2.npz")      # training only: not needed online
    citizen = clip(hello, "asl_citizen", "landmarks/asl_citizen/hello.npz", canonical=True)
    own = clip(mine, "user", "landmarks/user/me/my-sign.npz", canonical=True)
    media(ReferenceVideo, a, commons, "videos/1/wikimedia_commons_A.mp4")
    media(ReferenceImage, a, photo, "images/1/kaggle_asl_alphabet_A1.jpg")
    media(ReferenceVideo, hello, citizen, "videos/2/asl_citizen_1.mp4")
    media(ReferenceVideo, mine, own, "videos/3/user_1.webm")
    src.add(Attempt(sign_id=a.id, score=50, learner_id="learner-1"))
    src.commit()

    n = publish(src, dst, src_store, dst_store)
    assert n == {"signs": 2, "clips": 2, "videos": 1, "images": 1, "files_uploaded": 4}
    assert sorted(dst_store.client.objects) == [
        "landmarks/kaggle_asl_alphabet/A1.npz", "landmarks/wikimedia_commons/A.npz",
        "media/images/1/kaggle_asl_alphabet_A1.jpg", "media/videos/1/wikimedia_commons_A.mp4"]
    signs = {s.label: s for s in dst.query(Sign)}
    assert set(signs) == {"A", "hello"} and signs["hello"].gloss == "HELLO"
    video, image = dst.query(ReferenceVideo).one(), dst.query(ReferenceImage).one()
    assert video.sign_id == image.sign_id == signs["A"].id
    assert dst.get(ReferenceClip, video.clip_id).landmarks_path == "landmarks/wikimedia_commons/A.npz"
    assert dst.get(ReferenceClip, image.clip_id).landmarks_path == "landmarks/kaggle_asl_alphabet/A1.npz"
    assert all(c.video_path is None for c in dst.query(ReferenceClip))
    assert dst.query(Attempt).count() == 0

    # Publishing again updates in place and uploads nothing.
    assert publish(src, dst, src_store, dst_store) == {**n, "files_uploaded": 0}
    assert dst.query(Sign).count() == 2 and dst.query(ReferenceClip).count() == 2
