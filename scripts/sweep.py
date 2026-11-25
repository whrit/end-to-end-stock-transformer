#!/usr/bin/env -S uv run
"""Hyperparameter sweeps: write a grid of experiment configs, then rank the runs.

    # 9 configs: d_model x learning rate on the 2019-2022 tuning window, one seed
    uv run scripts/sweep.py make configs/us/stage1/master_d512_us_market63.json \\
        --prefix tune_us --out configs/us/secondary \\
        --fix research_position='"secondary"' --fix seeds='[42]' \\
        --fix rolling.test_start='"2019-01-01"' --fix rolling.test_end='"2022-12-31"' \\
        --grid model.d_model=128,256,512 --grid training.optimizer.learning_rate=1e-5,3e-5,1e-4

    # rank every finished tune_us_* run (local runs/ dir or r2://.../runs)
    uv run --env-file .env scripts/sweep.py compare r2://stock-transformer-data/us/v1/runs --prefix tune_us

    # more seeds of a config, as separate runs that train in parallel (<name>_seed43, ...)
    uv run scripts/sweep.py make configs/us/secondary/tune_v3_gru.json --prefix tune_v3_gru \\
        --out configs/us/secondary --grid seeds='[43]','[44]'

Values are JSON. Runs are ranked on the out-of-sample rank IC of their test
window (the tuning window, never the final test period); "valid rIC" is the
in-target validation score that early stopping already picked, so it is biased up.
compare groups ``<name>_seedNN`` runs with ``<name>``: rank IC is the mean over
all their seeds and "seed sd" its spread across seeds.
"""

import argparse
import itertools
import json
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from rich.console import Console
from rich.table import Table

from market_guided_stock_alpha import storage
from market_guided_stock_alpha.progress import run_main
from market_guided_stock_alpha.rolling import load_experiment_config

SHORT = {"d_model": "d", "learning_rate": "lr", "dropout": "drop", "name": "loss", "beta": "beta"}


def _assign(document: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = document
    for key in keys[:-1]:
        if key not in node:
            raise SystemExit("error: {} is not in the base config".format(dotted))
        node = node[key]
    if keys[-1] not in node:
        raise SystemExit("error: {} is not in the base config".format(dotted))
    node[keys[-1]] = value


def _pair(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected KEY=VALUE, got {}".format(text))
    key, value = text.split("=", 1)
    return key, value


def _label(dotted: str, value: Any) -> str:
    key = dotted.rsplit(".", 1)[-1]
    if key == "seeds" and isinstance(value, list) and len(value) == 1:
        return "seed{}".format(value[0])
    text = "{:g}".format(value) if isinstance(value, float) else str(value)
    return "{}{}".format(SHORT.get(key, key), text)


def _dump(document: Mapping[str, Any]) -> str:
    # One top-level key per line, like the hand-written configs.
    return "{\n" + ",\n".join('  "{}": {}'.format(k, json.dumps(v)) for k, v in document.items()) + "\n}\n"


def cmd_make(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    for key, value in args.fix:
        _assign(base, key, json.loads(value))
    axes = [(key, json.loads("[{}]".format(values))) for key, values in args.grid]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for combination in itertools.product(*[values for _key, values in axes]):
        document = json.loads(json.dumps(base))
        labels = []
        for (key, _values), value in zip(axes, combination):
            _assign(document, key, value)
            labels.append(_label(key, value))
        document["experiment_name"] = "_".join([args.prefix] + labels)
        path = out / "{}.json".format(document["experiment_name"])
        if path.exists() and not args.overwrite:
            raise SystemExit("error: {} exists (pass --overwrite to replace it)".format(path))
        path.write_text(_dump(document), encoding="utf-8")
        load_experiment_config(path)  # fail now, not on the GPU
        written.append(path)
    for path in written:
        print(path)
    return 0


def _runs(root: str, prefix: str, directory: Path) -> Path:
    """Local folder holding <run>/seed_*/... metric files for runs named prefix*."""
    if not storage.is_remote(root):
        return Path(root)

    def skip(relative: str) -> bool:
        parts = relative.split("/")
        return not (
            parts[0].startswith(prefix)
            and len(parts) > 2
            and parts[1].startswith("seed_")
            and parts[-1] in ("overall.json", "metrics.json")
        )

    storage.download_tree(root, directory, skip)
    return directory


_SEED_SUFFIX = re.compile(r"_seed\d+$")


def _summary(name: str, runs: Sequence[Path]) -> Dict[str, Any]:
    overall = [json.loads(path.read_text(encoding="utf-8")) for run in runs for path in sorted(run.glob("seed_*/overall.json"))]
    targets = [
        json.loads(path.read_text(encoding="utf-8")) for run in runs for path in sorted(run.glob("seed_*/target_*/metrics.json"))
    ]

    def mean(values: Sequence[Any]) -> float:
        numbers = [float(value) for value in values if value is not None]
        return sum(numbers) / len(numbers) if numbers else float("nan")

    rank_ics = [float(item["rank_ic"]) for item in overall if item.get("rank_ic") is not None]
    training = [target.get("training") or {} for target in targets]
    return {
        "run": name,
        "seeds": len(overall),
        "targets": len(targets),
        "rank_ic": mean(rank_ics),
        "seed_sd": statistics.stdev(rank_ics) if len(rank_ics) > 1 else float("nan"),
        "ic": mean([item.get("ic") for item in overall]),
        "top10": mean([item.get("top10") for item in overall]),
        "valid": mean([item.get("best_validation_metric") for item in training]),
        "epochs": mean([len(item["history"]) for item in training if isinstance(item.get("history"), list)]),
        "best_epoch": mean([item.get("best_epoch") for item in training]),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="sweep_") as directory:
        root = _runs(args.root, args.prefix, Path(directory))
        groups: Dict[str, List[Path]] = {}
        for run in sorted(root.iterdir()):
            if run.is_dir() and run.name.startswith(args.prefix):
                groups.setdefault(_SEED_SUFFIX.sub("", run.name), []).append(run)
        rows: List[Dict[str, Any]] = [_summary(name, runs) for name, runs in groups.items()]
    rows = [row for row in rows if row["seeds"]]
    if not rows:
        raise SystemExit("error: no finished seed (seed_*/overall.json) for runs named {}*".format(args.prefix))
    rows.sort(key=lambda row: row["rank_ic"], reverse=True)
    table = Table(title="{}* ranked by out-of-sample rank IC".format(args.prefix), header_style="bold dodger_blue1")
    for name in ("run", "seeds", "targets", "rank IC", "seed sd", "IC", "top10", "valid rIC", "epochs", "best ep"):
        table.add_column(name, justify="left" if name == "run" else "right", no_wrap=True)
    for row in rows:
        table.add_row(
            row["run"], str(row["seeds"]), str(row["targets"]),
            "{:+.4f}".format(row["rank_ic"]), "{:.4f}".format(row["seed_sd"]),
            "{:+.4f}".format(row["ic"]), "{:+.4f}".format(row["top10"]),
            "{:+.4f}".format(row["valid"]), "{:.1f}".format(row["epochs"]), "{:.1f}".format(row["best_epoch"]),
        )
    Console().print(table)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="write one config per grid point")
    make.add_argument("base", help="base experiment config")
    make.add_argument("--prefix", required=True, help="experiment_name prefix of every generated config")
    make.add_argument("--out", required=True, help="directory for the generated configs")
    make.add_argument("--fix", type=_pair, action="append", default=[], metavar="KEY=JSON", help="override in every config")
    make.add_argument("--grid", type=_pair, action="append", default=[], metavar="KEY=V1,V2", help="sweep axis (JSON values)")
    make.add_argument("--overwrite", action="store_true")
    make.set_defaults(run=cmd_make)
    compare = sub.add_parser("compare", help="rank finished runs")
    compare.add_argument("root", help="runs root: local dir or r2://.../runs")
    compare.add_argument("--prefix", required=True, help="experiment_name prefix to compare")
    compare.set_defaults(run=cmd_compare)
    args = parser.parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(run_main(main))
