#!/usr/bin/env -S uv run
"""Build the canonical relative-excess label from transparent tabular files."""

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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
from market_guided_stock_alpha.progress import board, kv_table, note, run_main
from market_guided_stock_alpha.storage import (
    is_remote,
    materialize,
    parse_r2_uri,
    publish,
)


def _display(location: Any) -> str:
    return str(location) if is_remote(location) else str(Path(location))


def _identity(location: Any) -> Any:
    if is_remote(location):
        return parse_r2_uri(location)
    return Path(location).resolve()


def _staging(location: Any, cache: Path, role: str) -> Path:
    if not is_remote(location):
        return Path(location)
    return cache / role / PurePosixPath(parse_r2_uri(location)[1]).name


def _source_metadata(location: Any, path: Path) -> Dict[str, Any]:
    stat = Path(path).stat()
    return {
        "path": _display(location),
        "sha256": sha256_path(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def default_manifest_location(output: Any) -> str:
    if is_remote(output):
        bucket, key = parse_r2_uri(output)
        return "r2://{}/{}".format(
            bucket, PurePosixPath(key).with_suffix(".manifest.json")
        )
    return str(Path(output).with_suffix(".manifest.json"))


def build_from_files(
    stock_path: Any,
    benchmark_path: Any,
    calendar_path: Any,
    output_path: Any,
    manifest_path: Any,
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    value_column: str = DEFAULT_VALUE_COLUMN,
    delistings_path: Any = None,
) -> Dict[str, Any]:
    """Load canonical files, build one artifact, and publish its manifest.

    Every location may be a local path or an ``r2://`` URI. ``delistings_path``
    (US v2 ``delistings.parquet``) adds delisting exits to the label.
    """
    sources = [
        ("stock_daily", stock_path),
        ("benchmark_daily", benchmark_path),
        ("calendar", calendar_path),
    ] + ([("delistings", delistings_path)] if delistings_path is not None else [])
    source_paths = {_identity(location) for _name, location in sources}
    if _identity(output_path) in source_paths:
        raise ValueError("Output path must not overwrite an input file")
    if _identity(manifest_path) in source_paths:
        raise ValueError("Manifest path must not overwrite an input file")
    if _identity(output_path) == _identity(manifest_path):
        raise ValueError("Output and manifest paths must be different")

    with tempfile.TemporaryDirectory(prefix="build_relative_excess_label_") as directory:
        cache = Path(directory)
        inputs = {
            name: (location, materialize(location, Path.cwd(), cache))
            for name, location in sources
        }
        local_output = _staging(output_path, cache, "output")
        local_manifest = _staging(manifest_path, cache, "manifest")
        note(
            ("━━ ", "bold magenta"),
            ("computing relative excess label", "bold"),
            (" (vectorized; no per-step progress)", "dim"),
        )
        output, diagnostics = build_relative_excess_label(
            read_table(inputs["stock_daily"][1]),
            read_table(inputs["benchmark_daily"][1]),
            read_table(inputs["calendar"][1]),
            benchmark_code=benchmark_code,
            value_column=value_column,
            delistings=(
                read_table(inputs["delistings"][1]) if "delistings" in inputs else None
            ),
        )
        output_sha256 = atomic_write_pickle(output, local_output)
        values = output[value_column].to_numpy(dtype="float64")
        builder_path = Path(__file__).resolve()
        formula_module = Path(relative_excess_return_module.__file__).resolve()
        manifest = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "formula": formula_metadata(benchmark_code, value_column),
            "inputs": {
                name: _source_metadata(location, path)
                for name, (location, path) in inputs.items()
            },
            "output": {
                "path": _display(output_path),
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
        atomic_write_json(local_manifest, manifest)
        publish(local_output, output_path)
        publish(local_manifest, manifest_path)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-daily", required=True)
    parser.add_argument("--benchmark-daily", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--benchmark-code", default=DEFAULT_BENCHMARK_CODE)
    parser.add_argument("--value-column", default=DEFAULT_VALUE_COLUMN)
    parser.add_argument(
        "--delistings", default=None,
        help="US v2 delistings.parquet: delisted stocks exit at their last close "
        "x exit_factor instead of losing the label",
    )
    args = parser.parse_args(argv)
    if PurePosixPath(args.output).suffix.lower() not in (".pkl", ".pickle"):
        parser.error("--output must use .pkl or .pickle")
    if args.manifest is None:
        args.manifest = default_manifest_location(args.output)
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
        delistings_path=args.delistings,
    )
    output = manifest["output"]
    quality = {
        "rows": output["row_count"],
        "dates": "{} → {} ({:,})".format(output["date_start"], output["date_end"], output["date_count"]),
        "securities": output["security_count"],
        "NaN values": output["nan_count"],
        "infinite values": output["infinite_count"],
        "duplicate index rows": output["duplicate_index_count"],
        "index sorted": output["index_sorted"],
    }
    for key, value in sorted(manifest.get("diagnostics", {}).items()):
        if isinstance(value, (int, float, str, bool)):
            quality[key.replace("_", " ")] = value
    board(kv_table(quality), "LABEL QUALITY")
    note(kv_table(quality, title="label quality"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(main))
