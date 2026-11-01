"""Where the store's files live: HANDSPEAK_STORE_DIR on local disk (the default) or an S3-compatible bucket such as
Cloudflare R2 (HANDSPEAK_STORAGE=s3, for hosts whose disk is wiped on every restart).

Keys are the paths the database holds, relative to the store dir: "landmarks/...", "recordings/..." and, for
reference videos and images (whose database paths are relative to media/), media_key(path) = "media/<path>". Both
backends use the same keys, so a store can be copied from one to the other as-is (scripts.publish_store).
"""
import mimetypes
from functools import lru_cache
from pathlib import Path, PurePosixPath

from .config import get_settings

URL_TTL_S = 6 * 3600       # lifetime of a signed media URL (S3)


def media_key(path):
    """Storage key of a reference_videos / reference_images path (None stays None)."""
    return f"media/{path}" if path else None


class LocalStorage:
    """Files under a directory."""
    kind = "local"

    def __init__(self, root):
        self.root = Path(root)

    def path(self, key) -> Path | None:
        """root/key, or None if the key is empty or leads out of root (a path is never followed out of the store)."""
        if not key:
            return None
        p = (self.root / key).resolve()
        return p if p.is_relative_to(self.root.resolve()) else None

    def exists(self, key) -> bool:
        p = self.path(key)
        return p is not None and p.is_file()

    def read(self, key) -> bytes:
        if not self.exists(key):
            raise FileNotFoundError(key)
        return self.path(key).read_bytes()

    def write(self, key, data: bytes):
        p = self.path(key)
        if p is None:
            raise ValueError(f"Invalid storage key: {key!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def delete(self, key) -> bool:
        if not self.exists(key):
            return False
        self.path(key).unlink()
        return True


def _object_key(key) -> str | None:
    """The key if it can't name anything outside the store (relative, '/'-separated, no '..'), else None."""
    if not key or key.startswith("/") or "\\" in key or ".." in PurePosixPath(key).parts:
        return None
    return key


def _checked(key) -> str:
    if _object_key(key) is None:
        raise ValueError(f"Invalid storage key: {key!r}")
    return key


def _missing(err) -> bool:
    return (getattr(err, "response", None) or {}).get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound")


class S3Storage:
    """Objects in an S3-compatible bucket. The bucket stays private: media is served through signed URLs."""
    kind = "s3"

    def __init__(self, bucket, endpoint=None, region=None, access_key_id=None, secret_access_key=None, client=None):
        if not bucket:
            raise ValueError("HANDSPEAK_STORAGE=s3 needs HANDSPEAK_S3_BUCKET (and, for R2, HANDSPEAK_S3_ENDPOINT).")
        if client is None:
            import boto3
            from botocore.config import Config
            # SigV4 signing, and checksums only where S3 requires them: R2 doesn't accept every checksum header that
            # newer boto3 versions add by default.
            client = boto3.client("s3", endpoint_url=endpoint, region_name=region or "auto",
                                  aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
                                  config=Config(signature_version="s3v4", request_checksum_calculation="when_required",
                                                response_checksum_validation="when_required"))
        self.bucket, self.client = bucket, client

    def exists(self, key) -> bool:
        if _object_key(key) is None:
            return False
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as e:  # noqa: BLE001 - botocore's ClientError; only "not found" means False
            if _missing(e):
                return False
            raise
        return True

    def read(self, key) -> bytes:
        if _object_key(key) is None:
            raise FileNotFoundError(key)
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:  # noqa: BLE001
            if _missing(e):
                raise FileNotFoundError(key) from None
            raise

    def write(self, key, data: bytes):
        ctype = mimetypes.guess_type(key)[0] or "application/octet-stream"
        self.client.put_object(Bucket=self.bucket, Key=_checked(key), Body=data, ContentType=ctype)

    def delete(self, key) -> bool:
        if not self.exists(key):
            return False
        self.client.delete_object(Bucket=self.bucket, Key=key)
        return True

    def url(self, key) -> str:
        """A signed GET URL for one object (browsers follow /media/... to it)."""
        return self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": _checked(key)},
                                                  ExpiresIn=URL_TTL_S)


@lru_cache
def get_storage():
    st = get_settings()
    if st.storage == "local":
        return LocalStorage(st.store_dir)
    if st.storage == "s3":
        return S3Storage(**st.s3)
    raise ValueError(f"HANDSPEAK_STORAGE must be 'local' or 's3', not {st.storage!r}.")
