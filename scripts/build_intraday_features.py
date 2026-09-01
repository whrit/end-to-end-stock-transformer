#!/usr/bin/env python3
"""Build canonical intraday feature artifacts from long or wide minute data."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


from market_guided_stock_alpha.artifacts import sha256_path
from market_guided_stock_alpha.data.contracts import MINUTE_FIELDS
from market_guided_stock_alpha.data.io import (
    read_table,
    read_wide_minute_pickles,
    wide_minute_frames_to_long,
)
from market_guided_stock_alpha.derived.builders import (
    build_intraday_market_artifact,
    build_intraday_price_volume_artifact,
)


BUILDERS = {
    "intraday_price_volume": (build_intraday_price_volume_artifact, "sec_code"),
    "intraday_market": (
        build_intraday_market_artifact,
        "index_code",
    ),
}


def _input_provenance(paths: Dict[str, Path]) -> Dict[str, Any]:
    return {
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_path(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in sorted(paths.items())
        },
        "builder": Path(__file__).name,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build stock price-volume or exploratory intraday market features from canonical "
            "long data or six aligned wide pandas pickles."
        )
    )
    parser.add_argument("family", choices=tuple(BUILDERS))
    parser.add_argument(
        "--input", type=Path,
        help="Canonical long CSV, Parquet, or pandas pickle table.",
    )
    for field in MINUTE_FIELDS:
        parser.add_argument(
            "--wide-{}".format(field), type=Path,
            help="Wide timestamp-by-instrument pandas pickle for {}.".format(field),
        )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="New immutable artifact directory; it must not already exist.",
    )
    return parser


def _load_input(args: argparse.Namespace, code_column: str):
    wide_paths = {
        field: getattr(args, "wide_{}".format(field)) for field in MINUTE_FIELDS
    }
    supplied_wide = {name: path for name, path in wide_paths.items() if path}
    if args.input is not None and supplied_wide:
        raise ValueError("use either --input or the six --wide-* files, not both")
    if args.input is not None:
        return read_table(args.input), {"long": args.input}
    if len(supplied_wide) != len(MINUTE_FIELDS):
        raise ValueError("provide --input or all six --wide-* pickle files")
    frames = read_wide_minute_pickles(supplied_wide)
    return wide_minute_frames_to_long(frames, code_column), supplied_wide


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    builder, code_column = BUILDERS[args.family]
    source, input_paths = _load_input(args, code_column)
    manifest = builder(
        source, args.output, provenance=_input_provenance(input_paths)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
