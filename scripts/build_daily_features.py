#!/usr/bin/env -S uv run
"""Build canonical daily feature artifacts from one transparent table."""

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Sequence


from market_guided_stock_alpha.artifacts import sha256_path
from market_guided_stock_alpha.derived.artifact import artifact_quality
from market_guided_stock_alpha.progress import board, kv_table, note, run_main
from market_guided_stock_alpha.data.io import read_table
from market_guided_stock_alpha.derived.builders import (
    ALPHA158_DTYPES,
    build_alpha158_artifact,
    build_market63_artifact,
    build_market84_artifact,
    build_us_market63_artifact,
    build_us_market84_artifact,
)
from market_guided_stock_alpha.storage import (
    is_remote,
    materialize,
    parse_r2_uri,
    publish,
)


BUILDERS = {
    "alpha158": build_alpha158_artifact,
    "market63": build_market63_artifact,
    "market84": build_market84_artifact,
    "us_market63": build_us_market63_artifact,
    "us_market84": build_us_market84_artifact,
}


def _input_provenance(location: str, path: Path) -> Dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "input": {
            "path": location if is_remote(location) else str(Path(location)),
            "sha256": sha256_path(source),
            "size_bytes": int(stat.st_size),
        },
        "builder": Path(__file__).name,
    }


def _local_output(location: str, cache: Path) -> Path:
    if not is_remote(location):
        return Path(location)
    return cache / "output" / PurePosixPath(parse_r2_uri(location)[1]).name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Alpha158, Market63, Market84, US Market63, or US Market84 "
            "from canonical daily data."
        )
    )
    parser.add_argument("family", choices=tuple(BUILDERS))
    parser.add_argument(
        "--input", required=True,
        help="Canonical CSV, Parquet, or pandas pickle table (path or r2:// URI).",
    )
    parser.add_argument(
        "--output", required=True,
        help=(
            "New immutable artifact directory (path or r2:// URI); "
            "it must not already exist."
        ),
    )
    parser.add_argument(
        "--dtype", choices=ALPHA158_DTYPES, default="float64",
        help="alpha158 only: storage dtype (float32 halves size; training uses float32).",
    )
    parser.add_argument(
        "--membership", default=None,
        help=(
            "alpha158 only: membership table (path or r2:// URI); securities that "
            "never appear in it are dropped. Their full history is kept otherwise."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help=(
            "alpha158 only: parallel worker processes over security batches "
            "(default: half the CPU cores). Workers run at low OS priority, so "
            "other apps stay responsive; output is identical for any value."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.family != "alpha158" and (args.dtype != "float64" or args.membership):
        parser.error("--dtype and --membership apply to alpha158 only")
    with tempfile.TemporaryDirectory(prefix="build_daily_features_") as directory:
        cache = Path(directory)
        source_path = materialize(args.input, Path.cwd(), cache)
        output = _local_output(args.output, cache)
        provenance = _input_provenance(args.input, source_path)
        options: Dict[str, Any] = {}
        if args.family == "alpha158":
            options["dtype"] = args.dtype
            options["workers"] = args.workers
            if args.membership:
                membership_path = materialize(args.membership, Path.cwd(), cache)
                provenance["membership"] = _input_provenance(
                    args.membership, membership_path
                )["input"]
                options["security_codes"] = (
                    read_table(membership_path)["sec_code"].unique().tolist()
                )
        manifest = BUILDERS[args.family](
            read_table(source_path), output, provenance=provenance, **options
        )
        quality = artifact_quality(output)
        board(kv_table(quality), "BUILD QUALITY")
        note(kv_table(quality, title="build quality"))
        publish(output, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(main))
