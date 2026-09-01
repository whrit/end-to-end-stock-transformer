#!/usr/bin/env python3
"""Build canonical daily feature artifacts from one transparent table."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


from market_guided_stock_alpha.artifacts import sha256_path
from market_guided_stock_alpha.data.io import read_table
from market_guided_stock_alpha.derived.builders import (
    build_alpha158_artifact,
    build_market63_artifact,
    build_market84_artifact,
)


BUILDERS = {
    "alpha158": build_alpha158_artifact,
    "market63": build_market63_artifact,
    "market84": build_market84_artifact,
}


def _input_provenance(path: Path) -> Dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "input": {
            "path": str(source),
            "sha256": sha256_path(source),
            "size_bytes": int(stat.st_size),
        },
        "builder": Path(__file__).name,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Alpha158, Market63, or Market84 from canonical daily data."
    )
    parser.add_argument("family", choices=tuple(BUILDERS))
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Canonical CSV, Parquet, or pandas pickle table.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="New immutable artifact directory; it must not already exist.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source = read_table(args.input)
    manifest = BUILDERS[args.family](
        source, args.output, provenance=_input_provenance(args.input)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
