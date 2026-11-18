"""R2 storage locations against an in-memory fake S3 client."""

import contextlib
import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha import storage
from market_guided_stock_alpha.derived import read_alpha158_artifact
from market_guided_stock_alpha.storage import (
    is_remote,
    materialize,
    parse_r2_uri,
    publish,
)


BUCKET = "test-bucket"


class FakeS3Client(object):
    """The three S3 client methods storage uses, backed by a dict."""

    def __init__(self, page_size=2):
        self.objects = {}
        self.page_size = page_size
        self.list_calls = 0

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        self.list_calls += 1
        keys = sorted(
            key for bucket, key in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        response = {"Contents": [{"Key": key} for key in page]}
        if start + self.page_size < len(keys):
            response["IsTruncated"] = True
            response["NextContinuationToken"] = str(start + self.page_size)
        else:
            response["IsTruncated"] = False
        return response

    def download_file(self, Bucket, Key, Filename):
        Path(Filename).write_bytes(self.objects[(Bucket, Key)])

    def upload_file(self, Filename, Bucket, Key):
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()


def daily_stock(periods=65):
    offset = np.arange(periods, dtype="float64")
    close = 20.0 + offset * 0.1
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-02", periods=periods),
            "sec_code": "000001.SZ",
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "vwap": close - 0.01,
            "volume": 1000.0 + offset,
        }
    )


def load_script(name):
    path = ROOT / "scripts" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location("storage_test_" + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParseR2UriTest(unittest.TestCase):
    def test_valid_uris(self):
        self.assertEqual(
            parse_r2_uri("r2://test-bucket/us/v1/raw/a.parquet"),
            ("test-bucket", "us/v1/raw/a.parquet"),
        )
        self.assertEqual(
            parse_r2_uri("r2://test-bucket/us/v1/"), ("test-bucket", "us/v1")
        )
        self.assertTrue(is_remote("r2://test-bucket/a"))
        self.assertFalse(is_remote(Path("data/a.parquet")))

    def test_invalid_uris(self):
        for uri in (
            "s3://test-bucket/a",
            "r2://test-bucket",
            "r2://test-bucket/",
            "r2://Bad_Bucket/a",
            "r2:///a",
            "r2://test-bucket//a",
            "r2://test-bucket/a/../b",
            "r2://test-bucket/./a",
        ):
            with self.subTest(uri=uri):
                with self.assertRaises(ValueError):
                    parse_r2_uri(uri)

    def test_missing_env_names_every_variable(self):
        with mock.patch.dict("os.environ", {"R2_ACCOUNT_ID": "acct"}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError, "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
            ):
                storage.r2_client_from_env()


class MaterializeTest(unittest.TestCase):
    def test_local_relative_and_absolute(self):
        base = Path("/configs/us")
        self.assertEqual(
            materialize("../data/a.csv", base, Path("/unused")),
            base / "../data/a.csv",
        )
        self.assertEqual(
            materialize(Path("/abs/a.csv"), base, Path("/unused")),
            Path("/abs/a.csv"),
        )

    def test_local_never_builds_client(self):
        with mock.patch.object(
            storage, "r2_client_from_env", side_effect=AssertionError("remote")
        ):
            materialize("a.csv", Path("/base"), Path("/unused"))
            publish(Path("/unused"), "a.csv")

    def test_remote_single_object(self):
        client = FakeS3Client()
        client.objects[(BUCKET, "us/v1/raw/a.csv")] = b"file"
        client.objects[(BUCKET, "us/v1/raw/a.csv.manifest.json")] = b"other"
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            path = materialize(
                "r2://test-bucket/us/v1/raw/a.csv", Path("/base"), cache, client
            )
            self.assertEqual(path, cache / BUCKET / "us" / "v1" / "raw" / "a.csv")
            self.assertEqual(path.read_bytes(), b"file")
            self.assertFalse((path.parent / "a.csv.manifest.json").exists())

    def test_remote_directory_prefix_with_pagination(self):
        client = FakeS3Client(page_size=2)
        client.objects[(BUCKET, "d/art/manifest.json")] = b"m"
        client.objects[(BUCKET, "d/art/values.npy")] = b"v"
        client.objects[(BUCKET, "d/art/nested/deep/x.npy")] = b"x"
        client.objects[(BUCKET, "d/art2/other.npy")] = b"o"
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            path = materialize("r2://test-bucket/d/art", Path("/base"), cache, client)
            self.assertEqual(path, cache / BUCKET / "d" / "art")
            self.assertEqual(
                sorted(
                    item.relative_to(path).as_posix()
                    for item in path.rglob("*") if item.is_file()
                ),
                ["manifest.json", "nested/deep/x.npy", "values.npy"],
            )
            self.assertEqual((path / "nested" / "deep" / "x.npy").read_bytes(), b"x")
            self.assertFalse((cache / BUCKET / "d" / "art2").exists())
            self.assertGreater(client.list_calls, 1)

    def test_remote_missing_raises(self):
        client = FakeS3Client()
        client.objects[(BUCKET, "d/artifact/x")] = b"x"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "r2://test-bucket/d/art"):
                materialize("r2://test-bucket/d/art", Path("/base"), Path(directory), client)

    def test_remote_traversal_key_rejected(self):
        client = FakeS3Client()
        client.objects[(BUCKET, "d/art/../../escape")] = b"x"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                materialize("r2://test-bucket/d/art", Path("/base"), Path(directory), client)
            with self.assertRaises(ValueError):
                materialize(
                    "r2://test-bucket/d/../escape", Path("/base"), Path(directory), client
                )


class PublishTest(unittest.TestCase):
    def test_file_and_directory_round_trip(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = root / "label.pkl"
            single.write_bytes(b"label")
            publish(single, "r2://test-bucket/us/v1/derived/label.pkl", client)

            artifact = root / "alpha158"
            (artifact / "nested").mkdir(parents=True)
            (artifact / "manifest.json").write_bytes(b"m")
            (artifact / "nested" / "values.npy").write_bytes(b"v")
            publish(artifact, "r2://test-bucket/us/v1/derived/alpha158/", client)

            self.assertEqual(
                sorted(key for _, key in client.objects),
                [
                    "us/v1/derived/alpha158/manifest.json",
                    "us/v1/derived/alpha158/nested/values.npy",
                    "us/v1/derived/label.pkl",
                ],
            )
            cache = root / "cache"
            copied = materialize(
                "r2://test-bucket/us/v1/derived/alpha158", root, cache, client
            )
            self.assertEqual((copied / "nested" / "values.npy").read_bytes(), b"v")
            self.assertEqual(
                materialize(
                    "r2://test-bucket/us/v1/derived/label.pkl", root, cache, client
                ).read_bytes(),
                b"label",
            )

    def test_existing_remote_location_is_immutable_unless_overwrite(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "alpha158"
            artifact.mkdir()
            (artifact / "manifest.json").write_bytes(b"m")
            publish(artifact, "r2://test-bucket/derived/alpha158", client)
            # A sibling sharing the string prefix is a different location.
            publish(artifact, "r2://test-bucket/derived/alpha158x", client)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish(artifact, "r2://test-bucket/derived/alpha158", client)
            (artifact / "manifest.json").write_bytes(b"new")
            publish(artifact, "r2://test-bucket/derived/alpha158", client, overwrite=True)
        self.assertEqual(
            client.objects[("test-bucket", "derived/alpha158/manifest.json")], b"new"
        )

    def test_missing_and_empty_sources_rejected(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                publish(Path(directory) / "absent", "r2://test-bucket/a", client)
            with self.assertRaisesRegex(ValueError, "empty directory"):
                publish(Path(directory), "r2://test-bucket/a", client)
        self.assertEqual(client.objects, {})

    def test_traversal_destination_rejected(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "a"
            source.write_bytes(b"a")
            with self.assertRaises(ValueError):
                publish(source, "r2://test-bucket/x/../../a", client)
        self.assertEqual(client.objects, {})


class ScriptRemoteTest(unittest.TestCase):
    def test_build_daily_features_reads_and_writes_r2(self):
        script = load_script("build_daily_features")
        client = FakeS3Client(page_size=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "daily_stock.pkl"
            daily_stock().to_pickle(source)
            payload = source.read_bytes()
            client.objects[(BUCKET, "us/v1/raw/daily_stock.pkl")] = payload
            with mock.patch.object(storage, "r2_client_from_env", return_value=client):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        script.main(
                            [
                                "alpha158",
                                "--input",
                                "r2://test-bucket/us/v1/raw/daily_stock.pkl",
                                "--output",
                                "r2://test-bucket/us/v1/derived/alpha158",
                            ]
                        ),
                        0,
                    )
            keys = sorted(
                key for _, key in client.objects
                if key.startswith("us/v1/derived/alpha158/")
            )
            self.assertIn("us/v1/derived/alpha158/manifest.json", keys)
            self.assertFalse(any(".partial" in key for key in keys))

            artifact_path = materialize(
                "r2://test-bucket/us/v1/derived/alpha158", root, root / "cache", client
            )
            artifact = read_alpha158_artifact(artifact_path)
            self.assertEqual(artifact.manifest["feature_count"], 158)
            provenance = artifact.manifest["provenance"]["input"]
            self.assertEqual(
                provenance["path"], "r2://test-bucket/us/v1/raw/daily_stock.pkl"
            )
            self.assertEqual(provenance["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(provenance["size_bytes"], len(payload))


if __name__ == "__main__":
    unittest.main()
