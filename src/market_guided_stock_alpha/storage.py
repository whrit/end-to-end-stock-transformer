"""Local-path or Cloudflare R2 (``r2://<bucket>/<key>``) artifact locations.

R2 is the only permanent store; jobs materialize inputs into a temporary local
cache and publish finished outputs back. Local locations keep the historical
path semantics and never import boto3.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

from market_guided_stock_alpha.progress import progress


R2_SCHEME = "r2://"
R2_ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def is_remote(location: Any) -> bool:
    return str(location).startswith(R2_SCHEME)


def _validate_key(key: str, uri: str) -> str:
    if not key or key.startswith("/"):
        raise ValueError("R2 key must be non-empty and relative: {}".format(uri))
    for segment in key.split("/"):
        if segment in ("", ".", ".."):
            raise ValueError(
                "R2 key has an empty, '.' or '..' segment: {}".format(uri)
            )
    return key


def parse_r2_uri(uri: str) -> Tuple[str, str]:
    text = str(uri)
    if not text.startswith(R2_SCHEME):
        raise ValueError("R2 URI must start with r2://: {}".format(text))
    bucket, separator, key = text[len(R2_SCHEME):].partition("/")
    if not _BUCKET_PATTERN.match(bucket):
        raise ValueError("R2 URI has an invalid bucket name: {}".format(text))
    if not separator:
        raise ValueError("R2 URI must name a key below the bucket: {}".format(text))
    if key.endswith("/"):
        key = key[:-1]
    return bucket, _validate_key(key, text)


def r2_client_from_env() -> Any:
    missing = [name for name in R2_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "r2:// locations need environment variables: {}".format(
                ", ".join(missing)
            )
        )
    try:
        import boto3
    except ImportError as error:
        raise ImportError("r2:// locations require boto3; install boto3") from error
    return boto3.client(
        "s3",
        endpoint_url="https://{}.r2.cloudflarestorage.com".format(
            os.environ["R2_ACCOUNT_ID"]
        ),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _list_keys(client: Any, bucket: str, prefix: str) -> Dict[str, int]:
    """Map every object key under ``prefix`` to its size in bytes."""
    keys: Dict[str, int] = {}
    request = {"Bucket": bucket, "Prefix": prefix}
    while True:
        response = client.list_objects_v2(**request)
        for item in response.get("Contents", ()):
            keys[item["Key"]] = int(item.get("Size", 0))
        if not response.get("IsTruncated"):
            return keys
        request["ContinuationToken"] = response["NextContinuationToken"]


def _byte_bar(action: str, uri: str, total: int) -> Any:
    return progress(
        total=total, desc="{} {}".format(action, uri.rstrip("/").rsplit("/", 1)[-1]), unit="B"
    )


def _download(client: Any, bucket: str, key: str, cache_dir: Path, bar: Any) -> Path:
    uri = "{}{}/{}".format(R2_SCHEME, bucket, key)
    target = Path(cache_dir, bucket, *_validate_key(key, uri).split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target), Callback=bar.update)
    return target


def materialize(
    location: Any, base: Path, cache_dir: Path, client: Any = None
) -> Path:
    """Return a local path for ``location``, downloading R2 objects if needed."""
    if not is_remote(location):
        path = Path(str(location))
        return path if path.is_absolute() else Path(base) / path
    bucket, key = parse_r2_uri(location)
    if client is None:
        client = r2_client_from_env()
    keys = _list_keys(client, bucket, key)
    if key in keys:
        with _byte_bar("download", str(location), keys[key]) as bar:
            return _download(client, bucket, key, cache_dir, bar)
    children = sorted(
        item for item in keys
        if item.startswith(key + "/") and not item.endswith("/")
    )
    if not children:
        raise FileNotFoundError("R2 location is missing: {}".format(location))
    with _byte_bar("download", str(location), sum(keys[c] for c in children)) as bar:
        for child in children:
            _download(client, bucket, child, cache_dir, bar)
    return Path(cache_dir, bucket, *key.split("/"))


def publish(
    local_path: Path, location: Any, client: Any = None, overwrite: bool = False
) -> None:
    """Upload a local file or directory to an R2 location; local is a no-op.

    Artifacts are immutable: an existing remote location is refused unless
    ``overwrite`` is set (rolling runs re-publish for ``--resume``).
    """
    if not is_remote(location):
        return
    bucket, key = parse_r2_uri(location)
    source = Path(local_path)
    if source.is_file():
        uploads = [(source, key)]
    elif source.is_dir():
        uploads = sorted(
            (
                (path, "{}/{}".format(key, path.relative_to(source).as_posix()))
                for path in source.rglob("*")
                if path.is_file()
            ),
            key=lambda item: item[1],
        )
        if not uploads:
            raise ValueError("Refusing to publish an empty directory: {}".format(source))
    else:
        raise FileNotFoundError("Local path to publish is missing: {}".format(source))
    if client is None:
        client = r2_client_from_env()
    if not overwrite and any(
        item == key or item.startswith(key + "/")
        for item in _list_keys(client, bucket, key)
    ):
        raise FileExistsError("R2 location already exists: {}".format(location))
    total = sum(path.stat().st_size for path, _ in uploads)
    with _byte_bar("upload", str(location), total) as bar:
        for path, object_key in uploads:
            client.upload_file(str(path), bucket, object_key, Callback=bar.update)
