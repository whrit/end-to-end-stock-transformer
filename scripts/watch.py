#!/usr/bin/env -S uv run
"""Watch a job's dashboard from any machine via its status snapshot.

Jobs write ``dashboard_status.json`` every 30 s: rolling runs into their run
directory automatically, other jobs when ``DASHBOARD_STATUS`` is set. Example:

    uv run --env-file .env scripts/watch.py \\
        r2://stock-transformer-data/us/v1/runs/<experiment>/dashboard_status.json
"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text

from market_guided_stock_alpha import progress, storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("status", help="dashboard_status.json location (local path or r2://)")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between polls (default 10)")
    parser.add_argument(
        "--exit-when-done", action="store_true", help="quit once the job reports a final status"
    )
    return parser


def _load(location: str, cache: Path) -> Dict[str, Any]:
    path = storage.materialize(location, Path.cwd(), cache, quiet=True)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = build_parser().parse_args()
    console = Console()
    with tempfile.TemporaryDirectory(prefix="watch_") as directory:
        cache = Path(directory)
        try:
            state: Dict[str, Any] = {"snapshot": _load(args.status, cache)}
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit("error: no readable status at {}: {}".format(args.status, error))
        if not console.is_terminal:
            snapshot = state["snapshot"]
            print("{} · {} · elapsed {:.0f}s".format(snapshot["title"], snapshot["status"], snapshot["elapsed"]))
            for stamp, markup, _detail in snapshot["tape"][-20:]:
                print(stamp, Text.from_markup(markup).plain)
            return 0

        def render() -> Any:
            width, height = console.size
            return progress.compose(progress.snapshot_view(state["snapshot"], width), width, height)

        last_error: Optional[str] = None
        progress.watch_keys(True)
        try:
            with Live(console=console, screen=True, refresh_per_second=2, get_renderable=render):
                next_poll = time.monotonic() + args.interval
                while True:
                    time.sleep(0.5)
                    if time.monotonic() < next_poll:
                        continue
                    next_poll = time.monotonic() + args.interval
                    try:
                        state["snapshot"] = _load(args.status, cache)
                        last_error = None
                    except Exception as error:  # keep showing the last snapshot
                        last_error = str(error)
                    if args.exit_when_done and state["snapshot"].get("status") != "RUNNING":
                        break
        except KeyboardInterrupt:
            pass
        finally:
            progress.watch_keys(False)
        snapshot = state["snapshot"]
        console.print(
            Text.assemble(
                ("watched {} · last status {}".format(snapshot["title"], snapshot["status"]), "bold"),
                ("  ({})".format(last_error) if last_error else "", "yellow"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
