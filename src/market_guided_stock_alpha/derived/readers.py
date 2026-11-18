"""Named thin readers over the common verified NPY/mmap artifact contract."""

from pathlib import Path
from typing import Optional

from market_guided_stock_alpha.derived.artifact import (
    FeatureArtifact,
    read_feature_artifact,
)


def read_alpha158_artifact(
    path: Path, verify_hashes: bool = True, mmap_mode: Optional[str] = "r"
) -> FeatureArtifact:
    return read_feature_artifact(path, "alpha158", verify_hashes, mmap_mode)


def read_market63_artifact(
    path: Path, verify_hashes: bool = True, mmap_mode: Optional[str] = "r"
) -> FeatureArtifact:
    return read_feature_artifact(path, "market63", verify_hashes, mmap_mode)


def read_market84_artifact(
    path: Path, verify_hashes: bool = True, mmap_mode: Optional[str] = "r"
) -> FeatureArtifact:
    return read_feature_artifact(path, "market84", verify_hashes, mmap_mode)


def read_us_market63_artifact(
    path: Path, verify_hashes: bool = True, mmap_mode: Optional[str] = "r"
) -> FeatureArtifact:
    return read_feature_artifact(path, "us_market63", verify_hashes, mmap_mode)


def read_us_market84_artifact(
    path: Path, verify_hashes: bool = True, mmap_mode: Optional[str] = "r"
) -> FeatureArtifact:
    return read_feature_artifact(path, "us_market84", verify_hashes, mmap_mode)


def read_intraday_price_volume_artifact(
    path: Path, verify_hashes: bool = True, mmap_mode: Optional[str] = "r"
) -> FeatureArtifact:
    return read_feature_artifact(path, "intraday_price_volume", verify_hashes, mmap_mode)


def read_intraday_market_artifact(
    path: Path, verify_hashes: bool = True, mmap_mode: Optional[str] = "r"
) -> FeatureArtifact:
    return read_feature_artifact(
        path, "intraday_market", verify_hashes, mmap_mode
    )
