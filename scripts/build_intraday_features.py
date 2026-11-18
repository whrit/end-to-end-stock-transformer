#!/usr/bin/env -S uv run
"""Build canonical intraday feature artifacts from long or wide minute data."""

import argparse
import json
import tempfile
from pathlib import Path, PurePosixPath
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
from market_guided_stock_alpha.storage import (
    is_remote,
    materialize,
    parse_r2_uri,
    publish,
)


BUILDERS = {
    "intraday_price_volume": (build_intraday_price_volume_artifact, "sec_code"),
    "intraday_market": (
        build_intraday_market_artifact,
        "index_code",
    ),
}


def _input_provenance(
    locations: Dict[str, str], paths: Dict[str, Path]
) -> Dict[str, Any]:
    return {
        "inputs": {
            name: {
                "path": (
                    locations[name]
                    if is_remote(locations[name])
                    else str(Path(locations[name]))
                ),
                "sha256": sha256_path(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in sorted(paths.items())
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
            "Build stock price-volume or exploratory intraday market features from canonical "
            "long data or six aligned wide pandas pickles."
        )
    )
    parser.add_argument("family", choices=tuple(BUILDERS))
    parser.add_argument(
        "--input",
        help="Canonical long CSV, Parquet, or pandas pickle table (path or r2:// URI).",
    )
    for field in MINUTE_FIELDS:
        parser.add_argument(
            "--wide-{}".format(field),
            help=(
                "Wide timestamp-by-instrument pandas pickle for {} "
                "(path or r2:// URI).".format(field)
            ),
        )
    parser.add_argument(
        "--output", required=True,
        help=(
            "New immutable artifact directory (path or r2:// URI); "
            "it must not already exist."
        ),
    )
    return parser


def _input_locations(args: argparse.Namespace) -> Dict[str, str]:
    wide_paths = {
        field: getattr(args, "wide_{}".format(field)) for field in MINUTE_FIELDS
    }
    supplied_wide = {name: path for name, path in wide_paths.items() if path}
    if args.input is not None and supplied_wide:
        raise ValueError("use either --input or the six --wide-* files, not both")
    if args.input is not None:
        return {"long": args.input}
    if len(supplied_wide) != len(MINUTE_FIELDS):
        raise ValueError("provide --input or all six --wide-* pickle files")
    return supplied_wide


def _load_input(
    locations: Dict[str, str], cache: Path, code_column: str
):
    paths = {
        name: materialize(location, Path.cwd(), cache)
        for name, location in locations.items()
    }
    if "long" in paths:
        return read_table(paths["long"]), paths
    frames = read_wide_minute_pickles(paths)
    return wide_minute_frames_to_long(frames, code_column), paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    builder, code_column = BUILDERS[args.family]
    locations = _input_locations(args)
    with tempfile.TemporaryDirectory(prefix="build_intraday_features_") as directory:
        cache = Path(directory)
        source, input_paths = _load_input(locations, cache, code_column)
        output = _local_output(args.output, cache)
        manifest = builder(
            source, output, provenance=_input_provenance(locations, input_paths)
        )
        publish(output, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
