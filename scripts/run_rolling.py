#!/usr/bin/env -S uv run
"""Run one canonical rolling experiment from a transparent JSON config."""

import argparse
from pathlib import Path

from market_guided_stock_alpha.rolling.configuration import run_from_json_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, predict, evaluate, and raw-ensemble a canonical rolling experiment."
    )
    parser.add_argument("config", type=Path, help="JSON experiment config")
    parser.add_argument("--resume", action="store_true", help="skip verified completed targets")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device: cpu, cuda, or mps (default: auto-select cuda > mps > cpu)",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_from_json_config(arguments.config, arguments.resume, arguments.device)
    print(result["run_directory"])


if __name__ == "__main__":
    main()
