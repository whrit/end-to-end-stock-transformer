"""Read and align an offline derived-label artifact without recomputation."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.artifacts import sha256_path
from market_guided_stock_alpha.labels.normalization import normalize_label_object


class DerivedLabelReader:
    """Load a derived artifact, preserve its provenance, and align membership."""

    def __init__(
        self, path: Path, column: str, manifest_path: Optional[Path] = None
    ) -> None:
        self.path = Path(path)
        self.column = str(column)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        if not self.column:
            raise ValueError("Derived-label value column must be non-empty")

    def _manifest_metadata(self, artifact_sha256: str) -> Dict[str, Any]:
        if self.manifest_path is None:
            return {
                "derived_label_manifest_path": None,
                "derived_label_manifest_sha256": None,
                "derived_label_formula": None,
                "derived_label_builder_sha256": None,
            }
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "Derived-label manifest is missing: {}".format(self.manifest_path)
            )
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        output_contract = document.get("output", {})
        declared_sha256 = output_contract.get("sha256")
        if declared_sha256 is not None and declared_sha256 != artifact_sha256:
            raise ValueError("Derived-label manifest artifact hash does not match")
        declared_column = output_contract.get("value_column")
        if declared_column is not None and declared_column != self.column:
            raise ValueError("Derived-label manifest value column does not match")
        implementation = document.get("implementation", {})
        return {
            "derived_label_manifest_path": str(self.manifest_path),
            "derived_label_manifest_sha256": sha256_path(self.manifest_path),
            "derived_label_formula": document.get("formula"),
            "derived_label_builder_sha256": implementation.get("builder_sha256"),
        }

    def load(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not self.path.is_file():
            raise FileNotFoundError(
                "Derived-label artifact is missing: {}".format(self.path)
            )
        before = self.path.stat()
        before_sha256 = sha256_path(self.path)
        source = pd.read_pickle(self.path)
        labels = normalize_label_object(source, self.column)
        if labels.duplicated(["trade_date", "sec_code"]).any():
            duplicate_count = int(
                labels.duplicated(
                    ["trade_date", "sec_code"], keep=False
                ).sum()
            )
            raise ValueError(
                "Derived-label artifact has {} duplicate keys".format(
                    duplicate_count
                )
            )
        after = self.path.stat()
        after_sha256 = sha256_path(self.path)
        unchanged = (
            before_sha256 == after_sha256
            and int(before.st_size) == int(after.st_size)
            and int(before.st_mtime_ns) == int(after.st_mtime_ns)
        )
        if not unchanged:
            raise RuntimeError("Derived-label artifact changed during read")
        metadata = {
            "raw_label_source": self.path.name,
            "raw_label_source_path": str(self.path),
            "raw_label_column": self.column,
            "source_sha256": before_sha256,
            "source_size_bytes": int(before.st_size),
            "source_mtime_ns": int(before.st_mtime_ns),
            "source_unchanged_during_read": True,
            "source_row_count": int(len(labels)),
            "source_finite_label_count": int(
                np.isfinite(labels["label_value"].to_numpy(dtype="float64")).sum()
            ),
        }
        metadata.update(self._manifest_metadata(before_sha256))
        return labels, metadata

    def align(self, target: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        required = {"trade_date", "sec_code"}
        if not required.issubset(target.columns):
            raise ValueError("Derived-label target requires trade_date and sec_code")
        labels, metadata = self.load()
        ordered = target.copy()
        ordered["_derived_label_row_order"] = np.arange(len(ordered), dtype="int64")
        merged = ordered.merge(
            labels,
            on=["trade_date", "sec_code"],
            how="left",
            validate="one_to_one",
            sort=False,
        ).sort_values("_derived_label_row_order", kind="mergesort")
        raw = merged["label_value"].to_numpy(dtype="float64")
        finite = np.isfinite(raw)
        merged["label_raw"] = np.where(finite, raw, np.nan).astype("float64")
        merged["label_mask"] = finite.astype("bool")
        merged = merged.drop(columns=["label_value", "_derived_label_row_order"])
        metadata.update(
            {
                "target_row_count": int(len(merged)),
                "aligned_finite_label_count": int(finite.sum()),
                "aligned_missing_label_count": int((~finite).sum()),
                "missing_label_policy": "masked_no_supervision",
                "label_extension": "none",
                "computed_label_count": 0,
            }
        )
        return merged.reset_index(drop=True), metadata
