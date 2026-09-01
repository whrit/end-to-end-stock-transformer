#!/usr/bin/env python3
"""Build the canonical relative-excess label from transparent tabular files."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

import market_guided_stock_alpha.labels.relative_excess_return as relative_excess_return_module
from market_guided_stock_alpha.artifacts import (
    atomic_write_json,
    atomic_write_pickle,
    sha256_path,
)
from market_guided_stock_alpha.data.io import read_table
from market_guided_stock_alpha.labels.relative_excess_return import (
    DEFAULT_BENCHMARK_CODE,
    DEFAULT_VALUE_COLUMN,
    build_relative_excess_label,
    formula_metadata,
)

def _source_metadata(path: Path) -> Dict[str, Any]:
    stat = Path(path).stat()
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_from_files(
    stock_path: Path,
    benchmark_path: Path,
    calendar_path: Path,
    output_path: Path,
    manifest_path: Path,
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    value_column: str = DEFAULT_VALUE_COLUMN,
) -> Dict[str, Any]:
    """Load canonical files, build one artifact, and publish its manifest."""
    source_paths = {
        Path(stock_path).resolve(),
        Path(benchmark_path).resolve(),
        Path(calendar_path).resolve(),
    }
    if Path(output_path).resolve() in source_paths:
        raise ValueError("Output path must not overwrite an input file")
    if Path(manifest_path).resolve() in source_paths:
        raise ValueError("Manifest path must not overwrite an input file")
    if Path(output_path).resolve() == Path(manifest_path).resolve():
        raise ValueError("Output and manifest paths must be different")

    output, diagnostics = build_relative_excess_label(
        read_table(stock_path),
        read_table(benchmark_path),
        read_table(calendar_path),
        benchmark_code=benchmark_code,
        value_column=value_column,
    )
    output_sha256 = atomic_write_pickle(output, output_path)
    values = output[value_column].to_numpy(dtype="float64")
    builder_path = Path(__file__).resolve()
    formula_module = Path(relative_excess_return_module.__file__).resolve()
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formula": formula_metadata(benchmark_code, value_column),
        "inputs": {
            "stock_daily": _source_metadata(stock_path),
            "benchmark_daily": _source_metadata(benchmark_path),
            "calendar": _source_metadata(calendar_path),
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "value_column": value_column,
            "row_count": int(len(output)),
            "date_count": (
                int(output.index.get_level_values("trade_date").nunique())
                if len(output)
                else 0
            ),
            "security_count": (
                int(output.index.get_level_values("sec_code").nunique())
                if len(output)
                else 0
            ),
            "date_start": (
                str(
                    pd.Timestamp(
                        output.index.get_level_values("trade_date").min()
                    ).date()
                )
                if len(output)
                else None
            ),
            "date_end": (
                str(
                    pd.Timestamp(
                        output.index.get_level_values("trade_date").max()
                    ).date()
                )
                if len(output)
                else None
            ),
            "nan_count": int(np.isnan(values).sum()),
            "infinite_count": int(np.isinf(values).sum()),
            "duplicate_index_count": int(output.index.duplicated().sum()),
            "index_sorted": bool(output.index.is_monotonic_increasing),
        },
        "diagnostics": diagnostics,
        "implementation": {
            "builder": "scripts/build_relative_excess_label.py",
            "builder_sha256": sha256_path(builder_path),
            "formula_module": (
                "market_guided_stock_alpha.labels.relative_excess_return"
            ),
            "formula_module_sha256": sha256_path(formula_module),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-daily", type=Path, required=True)
    parser.add_argument("--benchmark-daily", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--benchmark-code", default=DEFAULT_BENCHMARK_CODE)
    parser.add_argument("--value-column", default=DEFAULT_VALUE_COLUMN)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() not in (".pkl", ".pickle"):
        parser.error("--output must use .pkl or .pickle")
    if args.manifest is None:
        args.manifest = args.output.with_suffix(".manifest.json")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = build_from_files(
        args.stock_daily,
        args.benchmark_daily,
        args.calendar,
        args.output,
        args.manifest,
        benchmark_code=args.benchmark_code,
        value_column=args.value_column,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
