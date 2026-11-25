"""Pull-based remote job queue (R2 or a local directory) and the GPU agent.

Standalone on purpose (stdlib + boto3, no package imports): a GPU instance
downloads this single file and starts the agent before the repository exists
there. The laptop side is ``scripts/remote.py``.

Layout under the jobs root (``r2://bucket/jobs`` or a local directory)::

    <job>/job.json       request, written last by submit (its presence = queued;
                         "exclusive": true = run alone, e.g. LightGBM's ~45 GB RAM)
    <job>/code.tar.gz    working-tree snapshot (tracked + untracked, not ignored)
    <job>/config.json    experiment config
    <job>/claim.json     created atomically by the agent that takes the job
    <job>/state.json     running / done / failed / stopped / cancelled
    <job>/job.log        full stdout+stderr of the job (uploaded every 30 s)
    <job>/control.json   {"stop": true} = finish current target, save, exit
    _agents/<name>.json  agent heartbeat (GPU, state, current jobs)
    _agent/jobs.py       this file, as fetched by the instance start command

Run the agent on an instance:  python jobs.py agent --root r2://bucket/jobs [--slots 2]
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Runs inside the job's locked environment; its JSON goes to job.log and state.json.
ENVIRONMENT_REPORT = (
    "import json, platform, torch; print(json.dumps({"
    "'python': platform.python_version(), 'torch': torch.__version__, "
    "'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), "
    "'cuda_available': torch.cuda.is_available(), "
    "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
)
R2_SCHEME = "r2://"
POLL_SECONDS = 15.0
LOG_UPLOAD_SECONDS = 30.0
HEARTBEAT_SECONDS = 60.0
FINAL_STATES = ("done", "failed", "stopped", "cancelled")
# Jobs one agent runs side by side on each GPU. Each training job keeps its
# per-target row table on the GPU (~5.6 GB for a late US v3 target) plus a 6 GB
# training reserve, so a slot needs ~12 GB of GPU memory; more than 3 jobs per
# GPU only fight over CPU cores.
GIB_PER_SLOT = 12
MAX_SLOTS = 3


RETRY_ATTEMPTS = 10  # final log and state writes ride out ~5 minutes without R2
RETRY_SECONDS = 30.0


def _retry(call: Any, *args: Any, **kwargs: Any) -> Any:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return call(*args, **kwargs)
        except Exception:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_SECONDS)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _upload_tree(store: "JobStore", source: Path, location: str) -> None:
    """Copy a local directory to ``location`` (``r2://bucket/key`` or a local path)."""
    files = [path for path in sorted(source.rglob("*")) if path.is_file()]
    if not location.startswith(R2_SCHEME):
        for path in files:
            target = Path(location) / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(path), str(target))
        return
    bucket, _, prefix = location[len(R2_SCHEME):].partition("/")
    for path in files:
        key = "{}/{}".format(prefix.strip("/"), path.relative_to(source).as_posix())
        _retry(store.client.upload_file, str(path), bucket, key)


def _r2_client() -> Any:
    import boto3  # type: ignore[import-not-found]

    missing = [
        name for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError("missing R2 environment variables: {}".format(", ".join(missing)))
    return boto3.client(
        "s3",
        endpoint_url="https://{}.r2.cloudflarestorage.com".format(os.environ["R2_ACCOUNT_ID"]),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _error_code(error: Exception) -> str:
    return str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))


class JobStore(object):
    """Tiny object store over R2 or a local directory, keyed by relative path."""

    def __init__(self, root: str) -> None:
        self.root = str(root).rstrip("/")
        self.remote = self.root.startswith(R2_SCHEME)
        if self.remote:
            bucket, _, prefix = self.root[len(R2_SCHEME):].partition("/")
            self.bucket = bucket
            self.prefix = prefix.strip("/")
            self.client = _r2_client()
        else:
            self.base = Path(self.root)

    def location(self, relative: str) -> str:
        return "{}/{}".format(self.root, relative)

    def _key(self, relative: str) -> str:
        return "/".join(part for part in (self.prefix, relative) if part)

    def get_text(self, relative: str) -> Optional[str]:
        if not self.remote:
            path = self.base / relative
            return path.read_text(encoding="utf-8") if path.is_file() else None
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(relative))
        except Exception as error:
            if _error_code(error) in ("NoSuchKey", "404"):
                return None
            raise
        return response["Body"].read().decode("utf-8")

    def get_json(self, relative: str) -> Optional[Dict[str, Any]]:
        text = self.get_text(relative)
        return None if text is None else json.loads(text)

    def put_text(self, relative: str, text: str, exclusive: bool = False) -> bool:
        """Write ``relative``; with ``exclusive`` only if it does not exist (atomic)."""
        data = text.encode("utf-8")
        if not self.remote:
            path = self.base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if exclusive:
                try:
                    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    return False
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                return True
            partial = path.with_name(path.name + ".partial")
            partial.write_bytes(data)
            os.replace(str(partial), str(path))
            return True
        extra = {"IfNoneMatch": "*"} if exclusive else {}
        try:
            self.client.put_object(Bucket=self.bucket, Key=self._key(relative), Body=data, **extra)
        except Exception as error:
            if exclusive and _error_code(error) in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                return False
            raise
        return True

    def put_json(self, relative: str, document: Dict[str, Any], exclusive: bool = False) -> bool:
        return self.put_text(relative, json.dumps(document, indent=2, sort_keys=True), exclusive)

    def put_file(self, relative: str, path: Path) -> None:
        if not self.remote:
            target = self.base / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(path), str(target))
            return
        self.client.upload_file(str(path), self.bucket, self._key(relative))

    def get_file(self, relative: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.remote:
            shutil.copyfile(str(self.base / relative), str(path))
            return
        self.client.download_file(self.bucket, self._key(relative), str(path))

    def children(self, relative: str = "") -> List[str]:
        """Immediate sub-directory names under ``relative``, sorted."""
        if not self.remote:
            folder = self.base / relative
            return sorted(item.name for item in folder.iterdir() if item.is_dir()) if folder.is_dir() else []
        prefix = self._key(relative)
        prefix = prefix + "/" if prefix else ""
        names = []
        request = {"Bucket": self.bucket, "Prefix": prefix, "Delimiter": "/"}
        while True:
            response = self.client.list_objects_v2(**request)
            for item in response.get("CommonPrefixes", ()):
                names.append(item["Prefix"][len(prefix):].rstrip("/"))
            if not response.get("IsTruncated"):
                return sorted(names)
            request["ContinuationToken"] = response["NextContinuationToken"]

    def files(self, relative: str) -> List[str]:
        """Plain file names directly under ``relative``."""
        if not self.remote:
            folder = self.base / relative
            return sorted(item.name for item in folder.iterdir() if item.is_file()) if folder.is_dir() else []
        prefix = self._key(relative) + "/"
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter="/")
        return sorted(item["Key"][len(prefix):] for item in response.get("Contents", ()))


def gpu_name() -> str:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        output = ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "no NVIDIA GPU"
    return "{}× {}".format(len(lines), lines[0]) if len(lines) > 1 else lines[0]


def gpu_count() -> int:
    """NVIDIA GPUs on this machine (1 when there are none, so CPU agents still run jobs)."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        output = ""
    return max(1, len([line for line in output.splitlines() if line.strip()]))


def default_slots() -> int:
    """Jobs per GPU that fit the smallest GPU here: 24 GB -> 2, 16 GB -> 1 (1 without GPUs)."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
        mebibytes = [int(line) for line in output.split() if line.strip()]
    except (OSError, subprocess.SubprocessError, ValueError):
        mebibytes = []
    if not mebibytes:
        return 1
    return max(1, min(MAX_SLOTS, (min(mebibytes) + 512) // 1024 // GIB_PER_SLOT))


def job_command(job: Dict[str, Any]) -> List[str]:
    command = [
        "uv", "run", "--frozen", "--all-extras", "scripts/run_rolling.py", "job_config.json",
        "--device", str(job.get("device", "cuda")),
    ]
    if job.get("resume"):
        command.append("--resume")
    if job.get("compare"):
        command += ["--compare", str(job["compare"])]
    return command


class Agent(object):
    """Claims queued jobs and runs up to ``slots`` per GPU at once, each pinned to
    the least-busy GPU; never touches the instance."""

    def __init__(self, store: JobStore, name: str, workdir: Path, once: bool = False, slots: Optional[int] = None) -> None:
        slots = default_slots() if slots is None else slots
        if slots < 1:
            raise ValueError("slots must be at least 1")
        self.store = store
        self.name = name
        self.workdir = workdir
        self.once = once
        self.slots = slots
        self.gpu = gpu_name()
        self.gpus = gpu_count()
        self.lock = threading.Lock()
        self.running: Dict[str, Dict[str, Any]] = {}
        self.devices: Dict[str, int] = {}  # job -> GPU index it is pinned to
        self.threads: List[threading.Thread] = []
        self.claimed = False

    def _least_busy_gpu(self) -> int:
        """Call with ``self.lock`` held."""
        load = [0] * self.gpus
        for index in self.devices.values():
            load[index] += 1
        return load.index(min(load))

    @property
    def state(self) -> str:
        with self.lock:
            return "running" if self.running else "idle"

    @property
    def job(self) -> Optional[str]:
        with self.lock:
            return ", ".join(sorted(self.running)) or None

    def heartbeat(self) -> None:
        self.store.put_json(
            "_agents/{}.json".format(self.name),
            {
                "name": self.name,
                "gpu": self.gpu,
                "host": socket.gethostname(),
                "state": self.state,
                "job": self.job,
                "slots": self.slots,
                "gpus": self.gpus,
                "updated_at": now_iso(),
            },
        )

    def _heartbeat_loop(self) -> None:
        while True:
            try:
                self.heartbeat()
            except Exception as error:  # keep the agent alive through transient errors
                print("heartbeat failed: {}".format(error), file=sys.stderr, flush=True)
            time.sleep(HEARTBEAT_SECONDS)

    def salvage(self) -> None:
        """Upload the run folders of jobs this machine was running when it stopped
        (e.g. out of credit), mark them stopped, so ``submit --resume`` continues
        from their finished targets instead of starting over."""
        if not self.workdir.is_dir():
            return
        for folder in sorted(path for path in self.workdir.iterdir() if path.is_dir()):
            try:
                job = self.store.get_json("{}/job.json".format(folder.name))
                state = (self.store.get_json("{}/state.json".format(folder.name)) or {}).get("state")
                if job is None or state in FINAL_STATES or not job.get("run_location"):
                    continue
                runs = [
                    path for path in (folder / "tmp").rglob(str(job["experiment"]))
                    if path.is_dir() and (path / "config.json").is_file()
                ]
                for run in runs:
                    _upload_tree(self.store, run, str(job["run_location"]))
                self._state(folder.name, "stopped", ended_at=now_iso(), salvaged=bool(runs))
                print("job {} salvaged ({} run folder(s)); resubmit with --resume".format(folder.name, len(runs)), flush=True)
            except Exception as error:  # never block the agent on a rescue attempt
                print("job {} salvage failed: {}".format(folder.name, error), file=sys.stderr, flush=True)

    def run_forever(self) -> None:
        # Job folders hold a full locked environment (~5 GB); left over from a crash
        # or an older agent they fill the instance disk, so start from an empty workdir,
        # after saving whatever an interrupted job had already trained.
        self.salvage()
        shutil.rmtree(str(self.workdir), ignore_errors=True)
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        print("agent {} on {} watching {} ({} slots per GPU)".format(self.name, self.gpu, self.store.root, self.slots), flush=True)
        while True:
            self.threads = [thread for thread in self.threads if thread.is_alive()]
            if self.once and self.claimed and not self.threads:
                return
            claimed = False
            if self._has_room():
                try:
                    claimed = self.poll_once(self._spawn)
                except Exception as error:
                    print("poll failed: {}".format(error), file=sys.stderr, flush=True)
            if not claimed:  # claimed: look again at once to fill the next free slot
                time.sleep(POLL_SECONDS)

    def _has_room(self) -> bool:
        with self.lock:
            exclusive = any(job.get("exclusive") for job in self.running.values())
        return len(self.threads) < self.slots * self.gpus and not exclusive and not (self.once and self.claimed)

    def _spawn(self, job_id: str, job: Dict[str, Any]) -> None:
        thread = threading.Thread(target=self.run_job, args=(job_id, job), name=job_id, daemon=True)
        thread.start()
        self.threads.append(thread)

    def poll_once(self, run: Any = None) -> bool:
        """Claim the first claimable job and hand it to ``run`` (default: run it here)."""
        with self.lock:
            busy = bool(self.running)
        for job_id in self.store.children():
            if job_id.startswith("_") or self.store.get_text("{}/claim.json".format(job_id)) is not None:
                continue
            job = self.store.get_json("{}/job.json".format(job_id))
            if job is None:
                continue  # submit still uploading
            if job.get("exclusive") and busy:
                continue  # needs the machine to itself; later jobs may still fit a free slot
            claim = {"agent": self.name, "gpu": self.gpu, "claimed_at": now_iso()}
            if self.store.put_json("{}/claim.json".format(job_id), claim, exclusive=True):
                with self.lock:
                    self.running[job_id] = job
                    self.devices[job_id] = self._least_busy_gpu()
                self.claimed = True
                (run or self.run_job)(job_id, job)
                return True
        return False

    def _state(self, job_id: str, state: str, **fields: Any) -> None:
        document = {"state": state, "agent": self.name, "gpu": self.gpu, "updated_at": now_iso()}
        document.update(fields)
        self.store.put_json("{}/state.json".format(job_id), document)

    def run_job(self, job_id: str, job: Dict[str, Any]) -> None:
        with self.lock:
            self.running[job_id] = job
            if job_id not in self.devices:
                self.devices[job_id] = self._least_busy_gpu()
            device = self.devices[job_id]
        self.heartbeat()
        root = self.workdir / job_id
        log_path = root / "job.log"
        started = now_iso()
        exit_code = None
        process: Optional[subprocess.Popen] = None
        versions: Optional[Dict[str, Any]] = None
        try:
            self._state(job_id, "running", started_at=started)
            shutil.rmtree(str(root), ignore_errors=True)
            root.mkdir(parents=True)
            self.store.get_file("{}/code.tar.gz".format(job_id), root / "code.tar.gz")
            with tarfile.open(str(root / "code.tar.gz")) as archive:
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(str(root), filter="data")
                else:
                    archive.extractall(str(root))
            self.store.get_file("{}/config.json".format(job_id), root / "job_config.json")
            environment = dict(os.environ)
            environment["MGSA_CONTROL"] = self.store.location("{}/control.json".format(job_id))
            environment["PYTHONUNBUFFERED"] = "1"
            # Downloaded inputs (tens of GB) live under the job folder, so deleting it frees them.
            (root / "tmp").mkdir()
            environment["TMPDIR"] = str(root / "tmp")
            # Pinned to one GPU; jobs sharing it cap their epoch cache to their share (modeling/transport.py).
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["MGSA_GPU_JOBS"] = "1" if job.get("exclusive") else str(self.slots)
            environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            command = job_command(job)
            with open(str(log_path), "ab") as log:
                versions = self._environment_report(root, environment, log)
                self._state(job_id, "running", started_at=started, environment=versions, gpu_index=device)
                log.write("$ {}\n".format(" ".join(command)).encode("utf-8"))
                log.flush()
                process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT, env=environment)
                last_upload = time.monotonic()
                while process.poll() is None:
                    time.sleep(2)
                    if time.monotonic() - last_upload >= LOG_UPLOAD_SECONDS:
                        last_upload = time.monotonic()
                        log.flush()
                        try:
                            self.store.put_file("{}/job.log".format(job_id), log_path)
                        except Exception as error:  # a network blip must not abandon the job
                            print("job {} log upload failed: {}".format(job_id, error), file=sys.stderr, flush=True)
                exit_code = process.returncode
        except Exception as error:
            # The local disk may be the problem (full), so the error goes to the queue too.
            print("job {} agent error: {}".format(job_id, error), file=sys.stderr, flush=True)
            try:
                with open(str(log_path), "a", encoding="utf-8") as log:
                    log.write("agent error: {}\n".format(error))
            except OSError:
                self.store.put_text("{}/agent_error.txt".format(job_id), "agent error: {}\n".format(error))
        finally:
            try:
                if process is not None and process.poll() is None:
                    process.kill()  # never leave an untracked job holding the GPU and disk
                    process.wait()
                if log_path.exists():
                    _retry(self.store.put_file, "{}/job.log".format(job_id), log_path)
                # Exit codes from progress.run_main: 0 done, 3 stopped (resumable), else failed.
                state = {0: "done", 3: "stopped"}.get(exit_code, "failed")
                _retry(
                    self._state, job_id, state, started_at=started, ended_at=now_iso(), exit_code=exit_code,
                    environment=versions,
                )
                print("job {} {} (exit {})".format(job_id, state, exit_code), flush=True)
            finally:
                shutil.rmtree(str(root), ignore_errors=True)  # results are on R2; free the disk
                with self.lock:
                    self.running.pop(job_id, None)
                    self.devices.pop(job_id, None)
                self.heartbeat()

    @staticmethod
    def _environment_report(root: Path, environment: Dict[str, str], log: Any) -> Optional[Dict[str, Any]]:
        """Install the locked environment and log the torch/CUDA/GPU the job will use."""
        command = ["uv", "run", "--frozen", "--all-extras", "python", "-c", ENVIRONMENT_REPORT]
        log.write("$ environment check (uv.lock)\n".encode("utf-8"))
        log.flush()
        result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, env=environment)
        log.write((result.stdout + result.stderr).encode("utf-8"))
        log.flush()
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    break
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MGSA remote job agent")
    sub = parser.add_subparsers(dest="command", required=True)
    agent = sub.add_parser("agent", help="claim and run queued jobs on this machine")
    agent.add_argument("--root", default=os.environ.get("MGSA_JOBS"), help="jobs root (default $MGSA_JOBS)")
    agent.add_argument("--name", default=os.environ.get("MGSA_AGENT") or socket.gethostname())
    agent.add_argument("--workdir", type=Path, default=Path.home() / "mgsa_jobs")
    agent.add_argument("--once", action="store_true", help="exit after one job (testing)")
    agent.add_argument("--slots", type=int, help="jobs per GPU (default: GPU memory / {} GB, at most {})".format(GIB_PER_SLOT, MAX_SLOTS))
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("--root or MGSA_JOBS is required")
    Agent(JobStore(args.root), args.name, args.workdir, args.once, args.slots).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
