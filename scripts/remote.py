#!/usr/bin/env -S uv run
"""Send training runs to GPU instances you started yourself (e.g. vast.ai).

Jobs travel through a queue in R2; an agent on each instance claims and runs
them; results land in the config's r2:// run directory as usual. No ssh/rsync.

    remote.py rent                       pick a vast.ai offer interactively, rent it with the agent preinstalled
    remote.py instances                  your vast.ai instances and whether their agent is online
    remote.py setup                      upload the agent, print the start command (for manual instances)
    remote.py submit CONFIG [--resume] [--compare RUN] [--device cuda]
    remote.py list                       jobs and online agents
    remote.py watch JOB                  live dashboard (Ctrl-C detaches; the job keeps running)
    remote.py logs JOB [--follow]        full job output
    remote.py stop JOB                   finish the current target, save, exit (cancels if queued)
    remote.py pull JOB [--checkpoints]   download the run directory to ./runs/<experiment>

The queue root is $MGSA_JOBS (default r2://stock-transformer-data/jobs). The
vast.ai key comes from $VAST_API_KEY or the vastai CLI's saved key.
"""

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.prompt import Confirm, Prompt
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from market_guided_stock_alpha import jobs, storage
from market_guided_stock_alpha.progress import run_main

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS = "r2://stock-transformer-data/jobs"
AGENT_ONLINE_SECONDS = 180
console = Console()


def _age(stamp: Optional[str]) -> float:
    if not stamp:
        return float("inf")
    return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()


def _ago(stamp: Optional[str]) -> str:
    seconds = _age(stamp)
    if seconds == float("inf"):
        return "—"
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return "{:.0f}{} ago".format(seconds / size, unit)
    return "{:.0f}s ago".format(seconds)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=True).stdout


def snapshot(archive: Path) -> Dict[str, Any]:
    """Tar the working tree: tracked and untracked files git does not ignore."""
    names = [name for name in _git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0") if name]
    count = 0
    with tarfile.open(str(archive), "w:gz") as tar:
        for name in sorted(set(names)):
            path = ROOT / name
            if not path.is_file() or Path(name).name.startswith(".env"):
                continue  # deleted-but-tracked files; never ship secrets
            tar.add(str(path), arcname=name)
            count += 1
    return {
        "commit": _git("rev-parse", "HEAD").strip(),
        "dirty": bool(_git("status", "--porcelain").strip()),
        "files": count,
    }


def _locations(config: Dict[str, Any]) -> List[str]:
    data = config.get("data", {})
    values = [str(value) for key, value in data.items() if key != "label_column" and value is not None]
    return values + [str(config.get("output_root", "runs"))]


def _store(args: argparse.Namespace) -> jobs.JobStore:
    return jobs.JobStore(args.jobs)


def _job_state(store: jobs.JobStore, job_id: str) -> Dict[str, Any]:
    state = store.get_json("{}/state.json".format(job_id))
    if state is not None:
        return state
    claim = store.get_json("{}/claim.json".format(job_id))
    if claim is None:
        return {"state": "queued"}
    if claim.get("cancelled"):
        return {"state": "cancelled"}
    return {"state": "starting", "agent": claim.get("agent"), "gpu": claim.get("gpu")}


def _agents(store: jobs.JobStore) -> List[Dict[str, Any]]:
    found = []
    for name in store.files("_agents"):
        document = store.get_json("_agents/{}".format(name))
        if document is not None:
            found.append(document)
    return found


def _start_script(store: jobs.JobStore, name: str = "") -> str:
    """Instance on-start commands: install uv, fetch jobs.py from the queue, start the agent."""
    if store.remote:
        fetch = (
            'uv run --no-project --with boto3 python -c "import os,boto3;'
            "boto3.client('s3',endpoint_url='https://%s.r2.cloudflarestorage.com'%os.environ['R2_ACCOUNT_ID'],"
            "aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],"
            "region_name='auto').download_file('{}','{}','jobs.py')\"".format(
                store.bucket, "/".join(part for part in (store.prefix, "_agent/jobs.py") if part)
            )
        )
    else:
        fetch = "cp {} jobs.py".format(store.location("_agent/jobs.py"))
    return "\n".join(
        [
            "command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates)",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            'export PATH="$HOME/.local/bin:$PATH"',
            "mkdir -p /root/mgsa && cd /root/mgsa",
            fetch,
            "nohup uv run --no-project --with boto3 python jobs.py agent --root {}{} >> agent.log 2>&1 &".format(
                store.root, " --name {}".format(name) if name else ""
            ),
        ]
    )


def _upload_agent(store: jobs.JobStore) -> None:
    store.put_file("_agent/jobs.py", ROOT / "src" / "market_guided_stock_alpha" / "jobs.py")


def cmd_setup(args: argparse.Namespace) -> int:
    store = _store(args)
    _upload_agent(store)
    start = _start_script(store)
    console.print(Text("Agent uploaded to {}".format(store.location("_agent/jobs.py")), style="bold green"))
    console.print(
        Panel(
            "R2_ACCOUNT_ID=…\nR2_ACCESS_KEY_ID=…\nR2_SECRET_ACCESS_KEY=…",
            title="1 · environment variables for your vast.ai template (bucket-scoped R2 token recommended)",
            border_style="dodger_blue1",
        )
    )
    console.print(Panel(start, title="2 · on-start script (vast.ai template → On-start Script)", border_style="dodger_blue1"))
    console.print(
        "Image: [bold]{}[/] (no torch inside; the job installs the same torch + CUDA runtime as your uv.lock, "
        "so pick an image whose CUDA version is ≤ the host's). "
        "Re-run [bold]remote.py setup[/] after changing jobs.py; instances pick it up on their next start.".format(DEFAULT_IMAGE)
    )
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    store = _store(args)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    locations = _locations(config) + ([args.compare] if args.compare else [])
    if store.remote:
        local = [value for value in locations if not storage.is_remote(value)]
        if local:
            raise SystemExit(
                "error: remote jobs need r2:// data, output_root and --compare locations; local: {}".format(", ".join(local))
            )
    experiment = str(config["experiment_name"])
    run_location = "{}/{}".format(str(config.get("output_root", "runs")).rstrip("/"), experiment)
    job_id = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), secrets.token_hex(2))
    with tempfile.TemporaryDirectory(prefix="remote_submit_") as directory:
        archive = Path(directory) / "code.tar.gz"
        code = snapshot(archive)
        store.put_file("{}/code.tar.gz".format(job_id), archive)
        store.put_text("{}/config.json".format(job_id), config_path.read_text(encoding="utf-8"))
        size = archive.stat().st_size
    job = {
        "id": job_id,
        "experiment": experiment,
        "config": str(config_path),
        "device": args.device,
        "resume": bool(args.resume),
        "compare": args.compare,
        "run_location": run_location,
        "status_location": run_location + "/dashboard_status.json",
        "code": code,
        "submitted_at": jobs.now_iso(),
        "submitted_from": socket.gethostname(),
    }
    store.put_json("{}/job.json".format(job_id), job)  # last: its presence queues the job
    console.print(
        Text.assemble(
            ("queued ", "bold green"),
            (job_id, "bold"),
            ("  {} · commit {}{} · {} files, {:.1f} MB".format(
                experiment, code["commit"][:8], " + uncommitted edits" if code["dirty"] else "", code["files"], size / 1e6
            ), "dim"),
        )
    )
    online = [agent for agent in _agents(store) if _age(agent.get("updated_at")) < AGENT_ONLINE_SECONDS]
    if not online:
        console.print(Text("no agent online yet: the job starts when an instance with the agent comes up", style="yellow"))
    if args.no_watch:
        return 0
    return _watch(store, job_id)


def cmd_list(args: argparse.Namespace) -> int:
    store = _store(args)
    agents = Table(title="agents", title_style="bold dodger_blue1", header_style="bold dodger_blue1")
    for column in ("name", "gpu", "state", "job", "last seen"):
        agents.add_column(column, no_wrap=True)
    for agent in _agents(store):
        online = _age(agent.get("updated_at")) < AGENT_ONLINE_SECONDS
        agents.add_row(
            agent.get("name", "?"), agent.get("gpu", "?"),
            Text(agent.get("state", "?") if online else "offline", style="green" if online else "dim"),
            agent.get("job") or "", _ago(agent.get("updated_at")),
        )
    table = Table(title="jobs (newest first)", title_style="bold dodger_blue1", header_style="bold dodger_blue1")
    for column in ("job", "experiment", "state", "gpu", "submitted"):
        table.add_column(column, no_wrap=True)
    styles = {"done": "green", "failed": "red", "running": "bold yellow", "stopped": "yellow", "queued": "cyan"}
    for job_id in [name for name in store.children() if not name.startswith("_")][::-1][: args.limit]:
        job = store.get_json("{}/job.json".format(job_id)) or {}
        state = _job_state(store, job_id)
        table.add_row(
            job_id, job.get("experiment", "?"),
            Text(state["state"], style=styles.get(state["state"], "dim")),
            state.get("gpu", ""), _ago(job.get("submitted_at")),
        )
    console.print(agents)
    console.print(table)
    return 0


def _status_updated(location: str) -> Optional[float]:
    try:
        if storage.is_remote(location):
            with tempfile.TemporaryDirectory() as directory:
                text = storage.materialize(location, Path.cwd(), Path(directory), quiet=True).read_text(encoding="utf-8")
        else:
            text = Path(location).read_text(encoding="utf-8")
        return float(json.loads(text)["updated_at"])
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def _log_tail(store: jobs.JobStore, job_id: str, lines: int = 15) -> str:
    text = store.get_text("{}/job.log".format(job_id)) or ""
    return "\n".join(text.splitlines()[-lines:])


def _watch(store: jobs.JobStore, job_id: str) -> int:
    job = store.get_json("{}/job.json".format(job_id))
    if job is None:
        raise SystemExit("error: no job {} in {}".format(job_id, store.root))

    def waiting(state: Dict[str, Any]) -> Panel:
        head = Text.assemble(
            (job_id + "  ", "bold"), (state["state"], "bold yellow"),
            ("  {}".format(state.get("gpu", "")), "cyan"), ("  · Ctrl-C detaches", "dim"),
        )
        if state["state"] == "queued":
            online = [a for a in _agents(store) if _age(a.get("updated_at")) < AGENT_ONLINE_SECONDS]
            head.append("\nagents online: {}".format(", ".join("{} ({})".format(a["name"], a.get("gpu")) for a in online) or "none"), style="dim")
        return Panel(Group(head, Text(_log_tail(store, job_id), style="dim")), title=" job ", border_style="dodger_blue1")

    try:
        state = _job_state(store, job_id)
        with Live(waiting(state), console=console, refresh_per_second=1) as live:
            while True:
                state = _job_state(store, job_id)
                if state["state"] in jobs.FINAL_STATES:
                    break
                claimed = store.get_json("{}/claim.json".format(job_id)) or {}
                updated = _status_updated(job["status_location"])
                if state["state"] == "running" and updated is not None and updated >= time.time() - 120 and claimed:
                    break
                live.update(waiting(state))
                time.sleep(10)
        if state["state"] not in jobs.FINAL_STATES:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "watch.py"), job["status_location"], "--exit-when-done"])
            state = _job_state(store, job_id)
    except KeyboardInterrupt:
        console.print(Text("detached; {} keeps running".format(job_id), style="dim"))
        return 0
    style = {"done": "bold green", "failed": "bold red"}.get(state["state"], "bold yellow")
    console.print(Text.assemble(("job {} ".format(job_id), "bold"), (state["state"], style)))
    if state["state"] in ("failed", "cancelled"):
        console.print(Text(_log_tail(store, job_id, 25), style="dim"))
    if state["state"] in ("done", "stopped"):
        console.print(Text("results: {}   (remote.py pull {})".format(job["run_location"], job_id), style="dim"))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    return _watch(_store(args), args.job)


def cmd_logs(args: argparse.Namespace) -> int:
    store = _store(args)
    shown = 0
    while True:
        text = store.get_text("{}/job.log".format(args.job)) or ""
        sys.stdout.write(text[shown:])
        sys.stdout.flush()
        shown = len(text)
        if not args.follow or _job_state(store, args.job)["state"] in jobs.FINAL_STATES:
            return 0
        time.sleep(10)


def cmd_stop(args: argparse.Namespace) -> int:
    store = _store(args)
    if store.put_json("{}/claim.json".format(args.job), {"cancelled": True, "at": jobs.now_iso()}, exclusive=True):
        store.put_json("{}/state.json".format(args.job), {"state": "cancelled", "updated_at": jobs.now_iso()})
        console.print(Text("{} cancelled before it started".format(args.job), style="yellow"))
        return 0
    store.put_json("{}/control.json".format(args.job), {"stop": True, "requested_at": jobs.now_iso()})
    console.print(
        Text("stop requested: {} finishes its current target, saves, and exits (seen within a few seconds)".format(args.job), style="yellow")
    )
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    store = _store(args)
    job = store.get_json("{}/job.json".format(args.job))
    location = job["run_location"] if job else args.job
    destination = Path(args.dest) / location.rstrip("/").rsplit("/", 1)[-1]

    def skip(relative: str) -> bool:
        return not args.checkpoints and ("/checkpoints/" in "/" + relative or relative.endswith(".pt"))

    if storage.is_remote(location):
        written = storage.download_tree(location, destination, skip)
    else:
        shutil.copytree(location, str(destination), dirs_exist_ok=True, ignore=None if args.checkpoints else shutil.ignore_patterns("checkpoints", "*.pt"))
        written = [path for path in destination.rglob("*") if path.is_file()]
    console.print(Text("{} files → {}".format(len(written), destination), style="bold green"))
    return 0


# ------------------------------------------------------------------ vast.ai

VAST_API = "https://console.vast.ai/api/v0"
# Minimal NVIDIA image: no torch or CUDA libraries of its own (the job installs
# torch and its CUDA runtime from uv.lock), but it carries NVIDIA's container
# settings. Its NVIDIA_REQUIRE_CUDA must not exceed the host driver floor
# (MIN_HOST_CUDA), or containers refuse to start on allowed hosts.
DEFAULT_IMAGE = "nvidia/cuda:13.0.3-base-ubuntu24.04"
R2_VARIABLES = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")

# Hard rental limits: prompts and flags may tighten them, never loosen them.
MAX_PRICE_PER_HOUR = 0.50
MIN_NETWORK_MBPS = 750.0  # both upload and download
REGIONS = ("US", "CA", "MX")  # North America, by vast.ai country code
# Compatibility with the locked CUDA torch, two independent checks:
# - host driver: cuda_max_good (the vastai CLI's `cuda_vers` alias, the max CUDA
#   version the driver supports) >= the locked CUDA major. Send cuda_max_good:
#   the REST API does not apply `cuda_vers` (it returned CUDA 12.x hosts).
# - GPU architecture: CUDA 13 torch has no kernels below Turing (sm_75), and
#   CUDA 13 drivers still run on GTX 10xx / TITAN X, so the driver check alone
#   would admit GPUs the job cannot use.
MIN_COMPUTE_CAPABILITY = 750  # vast.ai encodes sm_75 as 750


def _locked_cuda_major() -> float:
    """CUDA major version of the torch runtime pinned in uv.lock (13.0 for 13.x)."""
    lines = (ROOT / "uv.lock").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() == 'name = "nvidia-cuda-runtime"' and lines[index + 1].startswith("version"):
            return float(lines[index + 1].split('"')[1].split(".")[0])
    return 12.0


MIN_HOST_CUDA = _locked_cuda_major()


def _image_cuda(image: str) -> Optional[float]:
    """CUDA major.minor of an ``nvidia/cuda:X.Y.Z-...`` image, else None."""
    if not image.startswith("nvidia/cuda:"):
        return None
    parts = image.split(":", 1)[1].split("-", 1)[0].split(".")
    try:
        return float("{}.{}".format(parts[0], parts[1] if len(parts) > 1 else 0))
    except ValueError:
        return None


def _country(offer: Dict[str, Any]) -> str:
    return str(offer.get("geolocation") or "").rsplit(",", 1)[-1].strip().upper()


def _within_limits(offer: Dict[str, Any], max_price: float, min_network: float) -> bool:
    return (
        float(offer.get("dph_total") or 0.0) <= max_price
        and float(offer.get("inet_up") or 0.0) >= min_network
        and float(offer.get("inet_down") or 0.0) >= min_network
        and _country(offer) in REGIONS
        and float(offer.get("compute_cap") or 0) >= MIN_COMPUTE_CAPABILITY
        and float(offer.get("cuda_max_good") or 0.0) >= MIN_HOST_CUDA
    )


def _checked_limits(max_price: Optional[float], min_network: Optional[float]) -> Tuple[float, float]:
    price = MAX_PRICE_PER_HOUR if max_price is None else float(max_price)
    network = MIN_NETWORK_MBPS if min_network is None else float(min_network)
    if not 0 < price <= MAX_PRICE_PER_HOUR:
        raise ValueError("max $/hour must be above 0 and at most {:.2f}".format(MAX_PRICE_PER_HOUR))
    if network < MIN_NETWORK_MBPS:
        raise ValueError("min network speed must be at least {:.0f} Mb/s".format(MIN_NETWORK_MBPS))
    return price, network


def _vast_key() -> str:
    key = os.environ.get("VAST_API_KEY", "").strip()
    for path in (Path.home() / ".config" / "vastai" / "vast_api_key", Path.home() / ".vast_api_key"):
        if not key and path.is_file():
            key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("error: set VAST_API_KEY in .env or run `vastai set api-key <key>`")
    return key


def _vast(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    request = urllib.request.Request(
        VAST_API + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        method=method,
        headers={
            "Authorization": "Bearer " + _vast_key(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit("error: vast.ai {} {} → HTTP {}: {}".format(method, path, error.code, detail))
    except urllib.error.URLError as error:
        raise SystemExit("error: cannot reach vast.ai: {}".format(error.reason))


def search_offers(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """On-demand North American offers within budget and network limits:
    the fastest (TFLOPS) offer at each price, cheapest first (``_best_value``)."""
    max_price, min_network = _checked_limits(filters["max_price"], filters["min_network"])
    query: Dict[str, Any] = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"gte": int(filters["gpus"])},
        "gpu_ram": {"gte": float(filters["gpu_ram"]) * 1024},
        "cpu_ram": {"gte": float(filters["ram"]) * 1024},
        "disk_space": {"gte": float(filters["disk"])},
        "reliability": {"gte": float(filters["reliability"])},
        "dph_total": {"lte": max_price},
        "inet_up": {"gte": min_network},
        "inet_down": {"gte": min_network},
        "geolocation": {"in": list(REGIONS)},
        "compute_cap": {"gte": MIN_COMPUTE_CAPABILITY},
        "cuda_max_good": {"gte": MIN_HOST_CUDA},  # = CLI `cuda_vers`
        "order": [["dph_total", "asc"]],
        "limit": 5000,  # a few hundred offers pass these limits; fetch them all
        "type": "on-demand",
    }
    if filters["verified"]:
        query["verified"] = {"eq": True}
    offers = _vast("POST", "/bundles/", query).get("offers", [])
    # Re-check client-side: never list an offer outside the hard limits.
    offers = [offer for offer in offers if _within_limits(offer, max_price, min_network)]
    if filters["gpu"]:
        # Substring match: "4090", "3090", "a4000", "rtx_5080" all work.
        needle = str(filters["gpu"]).lower().replace("_", " ")
        offers = [offer for offer in offers if needle in str(offer.get("gpu_name", "")).lower()]
    return _best_value(offers)


def _best_value(offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cheapest first, keeping only offers with more TFLOPS per GPU than every
    cheaper (or equal-priced) one: each row is the most compute a single-GPU job
    can use at its price. Per GPU because ``run_rolling.py`` trains on one GPU."""
    ranked = sorted(offers, key=lambda offer: (float(offer.get("dph_total") or 0.0), -_tflops(offer)))
    kept: List[Dict[str, Any]] = []
    for offer in ranked:
        if not kept or _tflops(offer) > _tflops(kept[-1]):
            kept.append(offer)
    return kept


def _tflops(offer: Dict[str, Any]) -> float:
    """TFLOPS of one GPU of the offer (vast.ai's ``total_flops`` sums all GPUs)."""
    return float(offer.get("total_flops") or 0.0) / max(int(offer.get("num_gpus") or 1), 1)


def _offer_table(offers: List[Dict[str, Any]]) -> Table:
    table = Table(header_style="bold dodger_blue1", border_style="grey37", title_style="bold dodger_blue1")
    for name, justify in (
        ("#", "right"), ("GPU", "left"), ("VRAM", "right"), ("RAM", "right"), ("disk", "right"),
        ("$/h", "right"), ("TFLOPS/GPU", "right"), ("DLPerf", "right"), ("perf/$", "right"), ("reliab.", "right"),
        ("net ↑", "right"), ("net ↓", "right"), ("CUDA", "right"), ("location", "left"),
    ):
        table.add_column(name, justify=justify, no_wrap=True)
    for index, offer in enumerate(offers, start=1):
        price = float(offer.get("dph_total") or 0.0)
        perf = float(offer.get("dlperf") or 0.0)
        reliability = float(offer.get("reliability2", offer.get("reliability", 0.0)) or 0.0)
        table.add_row(
            str(index),
            Text("{}× {}".format(offer.get("num_gpus", 1), offer.get("gpu_name", "?")), style="bold"),
            "{:.0f} GB".format(float(offer.get("gpu_ram") or 0) / 1024),
            "{:.0f} GB".format(float(offer.get("cpu_ram") or 0) / 1024),
            "{:.0f} GB".format(float(offer.get("disk_space") or 0)),
            Text("${:.3f}".format(price), style="green"),
            Text("{:.1f}".format(_tflops(offer)), style="bold"),
            "{:.0f}".format(perf),
            Text("{:.0f}".format(perf / price if price else 0.0), style="cyan"),
            Text("{:.1%}".format(reliability), style="green" if reliability >= 0.99 else "yellow"),
            "{:.0f} Mb/s".format(float(offer.get("inet_up") or 0)),
            "{:.0f} Mb/s".format(float(offer.get("inet_down") or 0)),
            str(offer.get("cuda_max_good", "?")),
            str(offer.get("geolocation") or "?"),
        )
    return table


def _ask_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    def ask(label: str, key: str, cast: Any) -> None:
        while True:
            current = filters[key]
            default = "any" if current in (None, "") and key == "gpu" else str(current)
            answer = Prompt.ask(label, default=default).strip()
            try:
                value = None if key == "gpu" and answer.lower() in ("any", "") else cast(answer)
                if key in ("max_price", "min_network"):
                    _checked_limits(
                        value if key == "max_price" else filters["max_price"],
                        value if key == "min_network" else filters["min_network"],
                    )
            except ValueError as error:
                console.print(Text(str(error), style="yellow"))
                continue
            filters[key] = value
            return

    console.print(Text("Filters (Enter keeps the value in brackets)", style="bold dodger_blue1"))
    console.print(
        Text(
            "Always applied: ≤ ${:.2f}/h · ≥ {:.0f} Mb/s up and down · North America ({}) · "
            "Turing+ GPU with CUDA ≥ {:.0f} driver (locked torch) · cheapest first, slower-and-pricier offers hidden".format(
                MAX_PRICE_PER_HOUR, MIN_NETWORK_MBPS, ", ".join(REGIONS), MIN_HOST_CUDA
            ),
            style="dim",
        )
    )
    ask("GPU name contains (e.g. 3090, 4090, A100, H100; 'any')", "gpu", str)
    ask("min GPUs", "gpus", int)
    ask("min VRAM per GPU, GB", "gpu_ram", float)
    ask("max $/hour (≤ {:.2f})".format(MAX_PRICE_PER_HOUR), "max_price", float)
    ask("min network up/down, Mb/s (≥ {:.0f})".format(MIN_NETWORK_MBPS), "min_network", float)
    ask("min system RAM, GB", "ram", float)
    ask("min disk, GB", "disk", float)
    ask("min reliability (0-1)", "reliability", float)
    filters["verified"] = Confirm.ask("verified hosts only", default=bool(filters["verified"]))
    return filters


def _instances() -> List[Dict[str, Any]]:
    return list(_vast("GET", "/instances/?owner=me").get("instances", []))


def cmd_instances(args: argparse.Namespace) -> int:
    store = _store(args)
    agents = {agent.get("name"): agent for agent in _agents(store)}
    table = Table(title="your vast.ai instances", header_style="bold dodger_blue1", title_style="bold dodger_blue1")
    for column in ("id", "label", "GPU", "status", "$/h", "agent", "agent job"):
        table.add_column(column, no_wrap=True)
    for instance in _instances():
        agent = agents.get("vast-{}".format(instance.get("id")))
        online = agent is not None and _age(agent.get("updated_at")) < AGENT_ONLINE_SECONDS
        status = str(instance.get("actual_status") or instance.get("cur_state") or "?")
        table.add_row(
            str(instance.get("id")), str(instance.get("label") or ""),
            "{}× {}".format(instance.get("num_gpus", 1), instance.get("gpu_name", "?")),
            Text(status, style="green" if status == "running" else "yellow"),
            "${:.3f}".format(float(instance.get("dph_total") or 0.0)),
            Text(agent.get("state", "?") if online else ("offline" if agent else "not seen"), style="green" if online else "dim"),
            (agent or {}).get("job") or "",
        )
    console.print(table)
    return 0


def _wait_for_agent(store: jobs.JobStore, instance_id: Any, timeout: float = 1800.0) -> bool:
    """Show instance boot and agent start-up; True once the agent heartbeats."""
    name = "vast-{}".format(instance_id)
    started = time.monotonic()
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while time.monotonic() - started < timeout:
                instance = next((item for item in _instances() if item.get("id") == instance_id), {})
                agent = store.get_json("_agents/{}.json".format(name))
                if agent is not None and _age(agent.get("updated_at")) < AGENT_ONLINE_SECONDS:
                    live.update(Text("agent {} online on {}".format(name, agent.get("gpu")), style="bold green"))
                    return True
                live.update(
                    Panel(
                        Text.assemble(
                            ("instance {}  ".format(instance_id), "bold"),
                            (str(instance.get("actual_status") or "creating"), "bold yellow"),
                            ("  {}".format(str(instance.get("status_msg") or "")[:100]), "dim"),
                            ("\nwaiting for the agent (image pull, uv install) · {:.0f}s elapsed · "
                             "Ctrl-C stops waiting; the instance keeps running".format(time.monotonic() - started), "dim"),
                        ),
                        title=" starting ", border_style="dodger_blue1",
                    )
                )
                time.sleep(10)
    except KeyboardInterrupt:
        return False
    return False


def cmd_rent(args: argparse.Namespace) -> int:
    store = _store(args)
    if not store.remote:
        raise SystemExit("error: rent needs an r2:// queue (MGSA_JOBS); instances cannot see a local folder")
    missing = [name for name in R2_VARIABLES if not os.environ.get(name)]
    if missing:
        raise SystemExit("error: missing {} (run with --env-file .env)".format(", ".join(missing)))
    try:
        _checked_limits(args.max_price, args.min_network)
    except ValueError as error:
        raise SystemExit("error: {}".format(error))
    image_cuda = _image_cuda(args.image)
    if image_cuda is not None and image_cuda > MIN_HOST_CUDA:
        raise SystemExit(
            "error: image {} requires CUDA {} but hosts are only guaranteed CUDA {:.1f}".format(
                args.image, image_cuda, MIN_HOST_CUDA
            )
        )
    filters: Dict[str, Any] = {
        "gpu": args.gpu, "gpus": args.gpus, "gpu_ram": args.min_vram,
        "max_price": args.max_price, "min_network": args.min_network,
        "ram": args.min_ram, "disk": args.disk, "reliability": args.min_reliability,
        "verified": not args.unverified,
    }
    interactive = sys.stdin.isatty() and not args.yes
    while True:
        if interactive:
            filters = _ask_filters(filters)
        with console.status("searching vast.ai offers…"):
            offers = search_offers(filters)[: args.show]
        if not offers:
            console.print(Text("no offers match these filters", style="yellow"))
            if not interactive:
                return 1
            continue
        console.print(_offer_table(offers))
        if args.pick is not None:
            choice = str(args.pick)
        elif interactive:
            choice = Prompt.ask("rent which # ('r' refine filters, 'q' quit)", default="1").strip().lower()
        else:
            raise SystemExit("error: pass --pick N to choose an offer non-interactively")
        if choice == "q":
            return 0
        if choice == "r":
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(offers):
            console.print(Text("not a listed number", style="yellow"))
            if not interactive:
                return 1
            continue
        offer = offers[int(choice) - 1]
        break
    max_price, min_network = _checked_limits(filters["max_price"], filters["min_network"])
    if not _within_limits(offer, max_price, min_network):
        raise SystemExit("error: offer {} is outside the rental limits; not renting".format(offer.get("id")))
    summary = "{}× {} · ${:.3f}/h · {:.0f}/{:.0f} Mb/s up/down · {:.0f} GB disk · {}".format(
        offer.get("num_gpus", 1), offer.get("gpu_name"), float(offer.get("dph_total") or 0),
        float(offer.get("inet_up") or 0), float(offer.get("inet_down") or 0),
        float(filters["disk"]), offer.get("geolocation") or "?",
    )
    console.print(Text("This rents a paid instance that runs until you destroy it: " + summary, style="bold yellow"))
    if not args.yes and not Confirm.ask("rent it", default=False):
        return 0
    _upload_agent(store)
    payload = {
        "client_id": "me",
        "image": args.image,
        "disk": float(filters["disk"]),
        "label": args.label,
        "runtype": "ssh",
        "env": {name: os.environ[name] for name in R2_VARIABLES},
        "onstart": _start_script(store, name="vast-$CONTAINER_ID"),
    }
    result = _vast("PUT", "/asks/{}/".format(offer["id"]), payload)
    instance_id = result.get("new_contract")
    if not result.get("success", True) or instance_id is None:
        raise SystemExit("error: vast.ai did not create the instance: {}".format(json.dumps(result)[:300]))
    console.print(Text.assemble(("rented instance ", "bold green"), (str(instance_id), "bold"), ("  " + summary, "dim")))
    if not _wait_for_agent(store, instance_id):
        console.print(Text("not waiting any longer; check later with `remote.py instances`", style="dim"))
        return 0
    config = args.config
    if config is None and interactive:
        config = Prompt.ask("submit a config now (path, blank to skip)", default="").strip() or None
    if config is None:
        return 0
    return cmd_submit(argparse.Namespace(
        jobs=args.jobs, config=config, device="cuda", resume=args.resume, compare=args.compare, no_watch=False,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", default=os.environ.get("MGSA_JOBS", DEFAULT_JOBS), help="queue root (r2:// or local dir)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="upload the agent and print the instance start command").set_defaults(run=cmd_setup)
    rent = sub.add_parser("rent", help="pick and rent a vast.ai offer with the agent preinstalled")
    rent.add_argument("--gpu", default=None, help="GPU name contains, e.g. 3090, 4090, A100, H100")
    rent.add_argument("--gpus", type=int, default=1, help="minimum GPU count")
    rent.add_argument("--min-vram", type=float, default=16.0, help="minimum VRAM per GPU, GB")
    rent.add_argument(
        "--max-price", type=float, default=MAX_PRICE_PER_HOUR,
        help="maximum $/hour (default and hard cap {:.2f})".format(MAX_PRICE_PER_HOUR),
    )
    rent.add_argument(
        "--min-network", type=float, default=MIN_NETWORK_MBPS,
        help="minimum upload and download Mb/s (default and floor {:.0f})".format(MIN_NETWORK_MBPS),
    )
    rent.add_argument("--min-ram", type=float, default=32.0, help="minimum system RAM, GB")
    rent.add_argument("--disk", type=float, default=60.0, help="disk to allocate, GB (data + environment ≈ 15 GB)")
    rent.add_argument("--min-reliability", type=float, default=0.98)
    rent.add_argument("--unverified", action="store_true", help="include unverified hosts")
    rent.add_argument("--show", type=int, default=20, help="offers to list (cheapest first, most TFLOPS per price)")
    rent.add_argument("--pick", type=int, default=None, help="offer number to rent without prompting")
    rent.add_argument("--yes", action="store_true", help="no prompts (needs --pick); still prints what it rents")
    rent.add_argument("--image", default=DEFAULT_IMAGE)
    rent.add_argument("--label", default="mgsa-agent")
    rent.add_argument("--config", default=None, help="submit this config once the agent is online")
    rent.add_argument("--resume", action="store_true")
    rent.add_argument("--compare", default=None)
    rent.set_defaults(run=cmd_rent)
    sub.add_parser("instances", help="your vast.ai instances and their agents").set_defaults(run=cmd_instances)
    submit = sub.add_parser("submit", help="queue a rolling run for a GPU agent")
    submit.add_argument("config", help="experiment JSON config (r2:// data and output_root)")
    submit.add_argument("--device", default="cuda")
    submit.add_argument("--resume", action="store_true")
    submit.add_argument("--compare", default=None, help="reference run (r2://) for the Δ rank IC column")
    submit.add_argument("--no-watch", action="store_true", help="queue and return")
    submit.set_defaults(run=cmd_submit)
    listing = sub.add_parser("list", help="jobs and agents")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(run=cmd_list)
    for name, handler, text in (
        ("watch", cmd_watch, "live dashboard of a job"),
        ("stop", cmd_stop, "graceful stop (or cancel if queued)"),
    ):
        command = sub.add_parser(name, help=text)
        command.add_argument("job")
        command.set_defaults(run=handler)
    logs = sub.add_parser("logs", help="job output")
    logs.add_argument("job")
    logs.add_argument("--follow", "-f", action="store_true")
    logs.set_defaults(run=cmd_logs)
    pull = sub.add_parser("pull", help="download a job's run directory")
    pull.add_argument("job", help="job id or run location")
    pull.add_argument("--dest", default="runs")
    pull.add_argument("--checkpoints", action="store_true", help="include model checkpoints")
    pull.set_defaults(run=cmd_pull)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(run_main(main))
