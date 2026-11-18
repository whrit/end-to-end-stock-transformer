"""Trading-terminal style live dashboard for every long-running job (rich).

Every job (fetch, builds, training) reports through this module: ``progress``
bars, ``note`` lines, ``board`` tables, ``metric`` values, ``curves`` charts,
``ticker`` items and ``header`` fields. State is always collected; how it is
shown depends on where the job runs:

* Interactive terminal: one full-screen dashboard (header, PROGRESS, METRICS,
  TAPE, BOARD, EQUITY, SYSTEM, ticker). Keys: ``s`` stop after the current
  target (rolling runs), ``↑``/``↓``/PgUp/PgDn/``j``/``k`` scroll the tape,
  ``G``/End back to live, ``t`` all events vs. important only. Windows below
  130x35 get a compact layout. At exit the screen closes and a summary is
  printed into the scrollback; stdout written meanwhile is replayed after it.
* Anywhere else (pipes, log files, pytest): notes print as plain lines.

Optional sinks, for any job: ``log_to(path)`` / ``DASHBOARD_LOG`` appends every
tape line to a file; ``status_to(location)`` / ``DASHBOARD_STATUS`` writes a
JSON snapshot every 2 s (local path or ``r2://``) that ``scripts/watch.py``
renders on another machine. On macOS a notification is sent when a dashboard
job (or any job with ``DASHBOARD_NOTIFY=1``) of 60 s or more ends.

Environment: ``PROGRESS_DISABLE=1`` plain mode, ``NO_COLOR=1`` no colour,
``DASHBOARD_ACCENT=<rich colour>`` accent colour, ``DASHBOARD_NOTIFY=0|1``.
"""

import atexit
import io
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import psutil
from rich import box, filesize
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.errors import StyleSyntaxError
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text


def _theme() -> Tuple[str, str]:
    custom = os.environ.get("DASHBOARD_ACCENT", "").strip()
    if custom:
        try:
            Style.parse(custom)
            return custom, "bold black on " + custom
        except StyleSyntaxError:
            pass
    return "dodger_blue1", "bold white on dodger_blue3"


ACCENT, ACCENT_BAR = _theme()
BORDER = "grey37"
SPARK = "▁▂▃▄▅▆▇█"
COMPACT_WIDTH = 130
COMPACT_HEIGHT = 35
STATUS_INTERVAL_SECONDS = 2.0  # snapshot ≈ 8 kB; at 2 s ≈ 1.3 M R2 writes per month of non-stop training
NOTIFY_MIN_SECONDS = 60.0
KEY_HINTS = "s stop after target · ↑↓ scroll · G live · t filter · Ctrl-C abort"
# The watcher only views a job: Ctrl-C closes the view, and stopping is `remote.py stop JOB`.
WATCH_HINTS = "↑↓ scroll · G live · t filter · Ctrl-C quit (the job keeps running)"
STOPPED_EXIT_CODE = 3  # graceful stop: outputs saved, resumable with --resume
_DEPTH_STYLES = ("bold cyan", "magenta", "blue", "bright_black")

_console = Console(stderr=True, highlight=False)


class StopRequested(Exception):
    """Raised by a job that honoured the ``s`` key after finishing a unit."""


class _CountColumn(ProgressColumn):
    """``412/676 date`` or, for bytes, ``1.2/6.5 GB``."""

    def render(self, task: Task) -> Text:
        unit = task.fields.get("unit", "it")
        if unit == "B":
            done = filesize.decimal(int(task.completed))
            total = "?" if task.total is None else filesize.decimal(int(task.total))
            return Text("{}/{}".format(done, total), style="green")
        total = "?" if task.total is None else "{:,}".format(int(task.total))
        return Text("{:,}/{} {}".format(int(task.completed), total, unit), style="green")


class _RateColumn(ProgressColumn):
    """``21.5 date/s``, ``2.4 s/epoch`` or ``46.2 MB/s``."""

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        if not speed:
            return Text("")
        return Text(_rate(speed, task.fields.get("unit", "it")), style="cyan")


def _rate(speed: float, unit: str) -> str:
    if unit == "B":
        return "{}/s".format(filesize.decimal(int(speed)))
    if speed >= 1:
        return "{:.1f} {}/s".format(speed, unit)
    return "{:.1f} s/{}".format(1.0 / speed, unit)


_progress = Progress(
    SpinnerColumn("dots", style=ACCENT),
    TextColumn("{task.description}", table_column=Column(width=30, no_wrap=True, overflow="ellipsis")),
    BarColumn(bar_width=28, style="grey35", complete_style=ACCENT, finished_style="green"),
    TaskProgressColumn(),
    _CountColumn(),
    _RateColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TextColumn(
        "{task.fields[postfix]}",
        style="yellow",
        markup=False,
        table_column=Column(no_wrap=True, overflow="ellipsis"),
    ),
    console=_console,
    auto_refresh=False,
)

# Job state: always collected. Mutated from the job thread, read by the
# refresh/status/key threads; readers copy containers first.
_began = False
_finished = False
_started = 0.0
_started_wall = 0.0
_title = ""
_header: Dict[str, str] = {}
_tape: Deque[Tuple[str, Text, bool]] = deque(maxlen=2000)
_metrics: Dict[str, Dict[str, Any]] = {}
_board: Any = None
_board_title = "BOARD"
_equity: Optional[Tuple[str, List[Tuple[str, List[float]]], List[int]]] = None
_ticker: Dict[str, Text] = {}
_exit_prints: List[Any] = []
_summary = ""
_eta_deadline: Optional[float] = None
_stoppable = False
_stop = threading.Event()
_log_file: Optional[IO[str]] = None
_status_location: Optional[str] = None
_status_thread: Optional[threading.Thread] = None
_status_error_reported = False
_job_mode = False  # set by run_main: file sinks only for real CLI jobs, not library/test calls

# Display state: only while a terminal shows the dashboard.
_live: Optional[Live] = None
_was_live = False
_keys_done = threading.Event()
_tape_offset = 0
_important_only = False
_stdout_buffer: List[str] = []
_saved: Dict[str, Any] = {}


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=int(max(seconds, 0))))


def _plain(renderable: Any, width: int = 120) -> str:
    console = Console(file=io.StringIO(), width=width, color_system=None, highlight=False)
    console.print(renderable)
    return console.file.getvalue()  # type: ignore[attr-defined]


def _ansi(renderable: Any, width: int) -> str:
    console = Console(
        file=io.StringIO(), width=width, force_terminal=True, color_system="truecolor", highlight=False
    )
    console.print(renderable)
    return console.file.getvalue()  # type: ignore[attr-defined]


# ------------------------------------------------------------------ lifecycle


def _begin() -> bool:
    """Start collecting (once) and open the dashboard if possible; True if live."""
    global _began, _started, _started_wall, _title
    if not _began:
        _began = True
        _started = time.monotonic()
        _started_wall = time.time()
        _title = " ".join(
            [os.path.basename(sys.argv[0] or "python")]
            + [os.path.basename(a) for a in sys.argv[1:3]]
        )
        _saved["excepthook"] = sys.excepthook
        sys.excepthook = _excepthook
        atexit.register(_shutdown, "DONE")
        if os.environ.get("DASHBOARD_LOG"):
            _open_log(os.environ["DASHBOARD_LOG"])
        if os.environ.get("DASHBOARD_STATUS"):
            _start_status(os.environ["DASHBOARD_STATUS"])
        if _display_enabled():
            _open_display()
    return _live is not None


def _display_enabled() -> bool:
    return os.environ.get("PROGRESS_DISABLE", "") in ("", "0") and _console.is_terminal


def _open_display() -> None:
    global _live, _was_live
    psutil.cpu_percent(None)
    _console.file = sys.stderr  # pin before stderr is redirected into the tape
    _saved.update(showwarning=warnings.showwarning, stdout=sys.stdout, stderr=sys.stderr)
    warnings.showwarning = _show_warning
    if sys.stdout.isatty():
        sys.stdout = _StdoutBuffer()  # type: ignore[assignment]
    sys.stderr = _StderrToTape()  # type: ignore[assignment]
    _live = Live(
        console=_console,
        screen=True,
        refresh_per_second=4,
        redirect_stdout=False,
        redirect_stderr=False,
        get_renderable=_render_live,
    )
    _live.start()
    _was_live = True
    _start_keys()


def _close_display(status: str) -> None:
    global _live
    if _live is None:
        return
    _stop_keys()
    _live.stop()
    _live = None
    warnings.showwarning = _saved["showwarning"]
    sys.stdout = _saved["stdout"]
    sys.stderr = _saved["stderr"]
    style = {"DONE": "bold green", "FAILED": "bold red"}.get(status, "bold yellow")
    _console.print(
        Rule(
            Text.assemble(
                ("MGSA ▸ {} ".format(_title), "bold " + ACCENT),
                (status, style),
                (" · elapsed {}".format(_duration(time.monotonic() - _started)), "dim"),
            ),
            style=ACCENT,
        )
    )
    if _exit_prints:
        for renderable in _exit_prints:
            _console.print(renderable)
    elif _board is not None:
        _console.print(_board)
    if _equity is not None:
        _console.print(_panel(_Chart(_equity[1], _equity[2]), _equity[0]))
    recent = [entry for entry in list(_tape) if not entry[2]][-12:]
    if recent:
        _console.print(Text("recent tape", style="bold " + ACCENT))
        for stamp, line, _detail in recent:
            _console.print(Text.assemble((stamp + " ", "dim"), line))
    if _log_file is not None:
        _console.print(Text("full tape: {}".format(_log_file.name), style="dim"))
    _console.file = None  # type: ignore[assignment]
    if _stdout_buffer:
        sys.stdout.write("".join(_stdout_buffer))
        sys.stdout.flush()
        del _stdout_buffer[:]


def _shutdown(status: str) -> None:
    """Close the display, write the final status, notify. Runs once."""
    global _finished
    if _finished or not _began:
        return
    _finished = True
    if _log_file is not None:
        _log_line(_clock(), Text("job {} · elapsed {}".format(status, _duration(time.monotonic() - _started))))
    _close_display(status)
    _write_status(status)
    _notify(status)
    if _log_file is not None:
        _log_file.close()
    if "excepthook" in _saved:
        sys.excepthook = _saved["excepthook"]


def _excepthook(kind: Any, value: Any, traceback: Any) -> None:
    hook = _saved.get("excepthook", sys.__excepthook__)
    _shutdown("INTERRUPTED" if issubclass(kind, KeyboardInterrupt) else "FAILED")
    hook(kind, value, traceback)


def run_main(main: Callable[[], Any]) -> int:
    """Run a script's ``main`` so every exit path closes the dashboard first.

    A ``SystemExit`` message (``raise SystemExit("error: ...")``) then appears
    after the summary instead of vanishing with the full-screen view, and a
    graceful stop (``s``) exits ``STOPPED_EXIT_CODE`` (3) with a resume hint.
    """
    global _job_mode
    _job_mode = True
    try:
        result = main()
    except StopRequested as stop:
        _shutdown("STOPPED")
        _console.print(Text(str(stop), style="bold yellow"))
        return STOPPED_EXIT_CODE
    except SystemExit as exit_request:
        _shutdown("DONE" if exit_request.code in (None, 0) else "FAILED")
        raise
    _shutdown("DONE")
    return 0 if result is None else int(result)


def _notify(status: str) -> None:
    mode = os.environ.get("DASHBOARD_NOTIFY", "")
    if sys.platform != "darwin" or mode == "0" or "pytest" in sys.modules:
        return
    if not (_was_live or mode == "1"):
        return
    elapsed = time.monotonic() - _started
    if elapsed < NOTIFY_MIN_SECONDS:
        return
    message = "{} after {}{}".format(status, _duration(elapsed), " · " + _summary if _summary else "")
    script = 'display notification {} with title {} sound name "{}"'.format(
        json.dumps(message, ensure_ascii=False),
        json.dumps("MGSA ▸ " + _title, ensure_ascii=False),
        "Glass" if status == "DONE" else "Basso",
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=5, check=False, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        pass


class _StdoutBuffer(object):
    """Holds stdout (e.g. JSON manifests) until the full-screen view closes."""

    encoding = "utf-8"

    def write(self, text: str) -> int:
        _stdout_buffer.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class _StderrToTape(object):
    """Stray Python-level stderr writes become detail tape lines."""

    encoding = "utf-8"

    def __init__(self) -> None:
        self.partial = ""

    def write(self, text: str) -> int:
        lines = (self.partial + text).split("\n")
        self.partial = lines.pop()
        for line in lines:
            if line.strip():
                _append_tape(Text(line.rstrip(), style="dim"), detail=True)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _show_warning(message: Any, category: Any, filename: str, lineno: int, *_: Any) -> None:
    note(
        ("⚠ {}: ".format(category.__name__), "bold yellow"),
        (str(message).strip(), "yellow"),
        ("  ({}:{})".format(os.path.basename(filename), lineno), "dim"),
    )


# ---------------------------------------------------------------- keyboard


def _start_keys() -> None:
    try:
        import termios
        import tty
    except ImportError:  # not a POSIX terminal
        return
    if not sys.stdin.isatty():
        return
    descriptor = sys.stdin.fileno()
    _saved["termios"] = (descriptor, termios.tcgetattr(descriptor))
    tty.setcbreak(descriptor)  # keeps ISIG, so Ctrl-C still interrupts
    _keys_done.clear()
    threading.Thread(target=_key_loop, args=(descriptor,), daemon=True).start()


def _stop_keys() -> None:
    _keys_done.set()
    saved = _saved.pop("termios", None)
    if saved is not None:
        import termios

        termios.tcsetattr(saved[0], termios.TCSADRAIN, saved[1])


def _scroll(lines: int) -> None:
    global _tape_offset
    _tape_offset = max(0, _tape_offset + lines)


def _scroll_to_live() -> None:
    global _tape_offset
    _tape_offset = 0


def _toggle_filter() -> None:
    global _important_only, _tape_offset
    _important_only = not _important_only
    _tape_offset = 0


def _toggle_stop() -> None:
    if not _stoppable:
        note(("this job has no safe stop point; Ctrl-C aborts it", "yellow"))
        return
    if _stop.is_set():
        _stop.clear()
        note(("stop cancelled", "bold yellow"))
    else:
        _stop.set()
        note(("■ stop requested: finishing the current target, then saving", "bold yellow"))


_KEYS: Tuple[Tuple[bytes, Callable[[], None]], ...] = (
    (b"\x1b[A", lambda: _scroll(1)),
    (b"\x1b[B", lambda: _scroll(-1)),
    (b"\x1b[5~", lambda: _scroll(15)),
    (b"\x1b[6~", lambda: _scroll(-15)),
    (b"\x1b[F", _scroll_to_live),
    (b"k", lambda: _scroll(1)),
    (b"j", lambda: _scroll(-1)),
    (b"G", _scroll_to_live),
    (b"t", _toggle_filter),
    (b"s", _toggle_stop),
)


def _key_loop(descriptor: int) -> None:
    pending = b""
    while not _keys_done.is_set():
        ready, _, _ = select.select([descriptor], [], [], 0.2)
        if not ready:
            continue
        data = os.read(descriptor, 32)
        if not data:
            return
        pending += data
        while pending:
            for sequence, action in _KEYS:
                if pending.startswith(sequence):
                    action()
                    pending = pending[len(sequence):]
                    break
            else:
                pending = pending[1:]


# ---------------------------------------------------------------- rendering


def _panel(content: Any, title: str, subtitle: str = "") -> Panel:
    return Panel(
        content,
        title=Text(" {} ".format(title), style=ACCENT_BAR),
        title_align="left",
        subtitle=Text(subtitle, style="dim") if subtitle else None,
        subtitle_align="right",
        border_style=BORDER,
        box=box.SQUARE,
    )


def _header_bar(title: str, fields: Mapping[str, str], elapsed: float, eta: Optional[float], flag: str) -> Table:
    left = Text(" MGSA ▸ {}".format(title), style="bold")
    for key, value in fields.items():
        left.append("  │  {} {}".format(key, value))
    if flag:
        left.append("  {} ".format(flag), style="bold white on red")
    right = Text(
        "{}  │  elapsed {}{} ".format(
            _clock(), _duration(elapsed), "" if eta is None else "  ETA {}".format(_duration(eta))
        )
    )
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right")
    grid.add_row(left, right, style=ACCENT_BAR)
    return grid


def _sparkline(values: List[float], width: int = 24) -> Text:
    values = values[-width:]
    low, high = min(values), max(values)
    span = high - low
    chars = "".join(
        SPARK[int((value - low) / span * (len(SPARK) - 1)) if span else 3] for value in values
    )
    return Text(chars, style=ACCENT)


def _metrics_table(metrics: Mapping[str, Mapping[str, Any]]) -> Table:
    table = Table(box=None, expand=True, padding=(0, 1), header_style="dim")
    for name, justify in (("metric", "left"), ("last", "right"), ("chg", "right"), ("best", "right"), ("trend", "left")):
        table.add_column(name, justify=justify, no_wrap=True)
    for label, entry in metrics.items():
        values: List[float] = list(entry["values"])
        if not values:
            continue
        higher = entry["higher"]
        last = values[-1]
        change = Text("")
        if len(values) > 1:
            delta = last - values[-2]
            good = delta > 0 if higher else delta < 0
            change = Text(
                "{}{:.4f}".format("▲" if delta > 0 else "▼" if delta < 0 else "■", abs(delta)),
                style="green" if good else "red" if delta else "dim",
            )
        best_index = values.index(max(values) if higher else min(values))
        best = Text.assemble(
            ("{:.4f}".format(values[best_index]), "green"), (" #{}".format(best_index + 1), "dim")
        )
        table.add_row(label, "{:.4f}".format(last), change, best, _sparkline(values))
    return table


class _Chart(object):
    """Braille line charts that fill the available width (one per series)."""

    ROWS = 4
    _BITS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))

    def __init__(self, series: Sequence[Tuple[str, List[float]]], marks: Sequence[int]) -> None:
        self.series = [(label, list(values)) for label, values in series if values]
        self.marks = list(marks)

    def height(self) -> int:
        return len(self.series) * (self.ROWS + 1)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(10, options.max_width)
        for label, values in self.series:
            last = values[-1]
            color = "green" if last >= 0 else "red"
            yield Text.assemble(
                (label, "bold"),
                ("  last ", "dim"),
                ("{:+.4f}".format(last), "bold " + color),
                ("  min {:+.4f}  max {:+.4f}  n={}".format(min(values), max(values), len(values)), "dim"),
            )
            for row in self._rows(values, width, color):
                yield row

    def _rows(self, values: List[float], width: int, color: str) -> List[Text]:
        low, high = min(min(values), 0.0), max(max(values), 0.0)
        span = (high - low) or 1.0
        height = self.ROWS * 4
        points = width * 2
        count = len(values)

        def y_of(value: float) -> int:
            return (height - 1) - int(round((value - low) / span * (height - 1)))

        def sample(x: int) -> float:
            """Linear interpolation, so few points still draw a smooth line."""
            position = x * (count - 1) / max(points - 1, 1)
            left = int(position)
            right = min(left + 1, count - 1)
            return values[left] + (values[right] - values[left]) * (position - left)

        ys = [y_of(sample(x)) for x in range(points)]
        cells = [[0] * width for _ in range(self.ROWS)]
        previous = ys[0]
        for x, y in enumerate(ys):
            for dot in range(min(previous, y), max(previous, y) + 1):
                cells[dot // 4][x // 2] |= self._BITS[x % 2][dot % 4]
            previous = y
        zero = y_of(0.0)
        mark_columns = set(int(m * (points - 1) / max(count - 1, 1)) // 2 for m in self.marks if 0 < m < count)
        rows = []
        for r in range(self.ROWS):
            line = Text()
            for c in range(width):
                bits = cells[r][c]
                if bits:
                    line.append(chr(0x2800 + bits), style=color)
                elif c in mark_columns:
                    line.append("┊", style="grey35")
                elif zero // 4 == r:
                    line.append(chr(0x2800 + self._BITS[0][zero % 4] + self._BITS[1][zero % 4]), style="grey35")
                else:
                    line.append(" ")
            rows.append(line)
        return rows


def _tape_body(entries: Sequence[Tuple[str, Text, bool]], lines: int, offset: int, important_only: bool) -> Tuple[Group, str]:
    visible = [entry for entry in entries if not (important_only and entry[2])]
    lines = max(lines, 1)
    offset = min(offset, max(len(visible) - lines, 0))
    end = len(visible) - offset
    rows = []
    for stamp, line, _detail in visible[max(end - lines, 0):end]:
        row = Text.assemble((stamp + " ", "dim"), line)
        row.no_wrap = True
        row.overflow = "ellipsis"
        rows.append(row)
    title = "TAPE"
    if important_only:
        title += " · important only"
    if offset:
        title += " · scrolled ↑{} (G = live)".format(offset)
    return Group(*rows), title


def _binary(size: float) -> str:
    """Binary units (GiB), matching how macOS reports RAM."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return "{:.1f} {}".format(size, unit) if unit != "B" else "{:.0f} B".format(size)
        size /= 1024.0
    return ""


def _gauge(fraction: float, label: str) -> Text:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * 16))
    style = "green" if fraction < 0.6 else "yellow" if fraction < 0.85 else "red"
    return Text.assemble(("━" * filled, style), ("━" * (16 - filled), "grey30"), " " + label)


def _system_table() -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    cpu = psutil.cpu_percent(None)
    table.add_row("CPU", _gauge(cpu / 100.0, "{:.0f}%".format(cpu)))
    memory = psutil.virtual_memory()
    table.add_row(
        "RAM",
        _gauge(memory.percent / 100.0, "{}/{}".format(_binary(memory.used), _binary(memory.total))),
    )
    try:
        process = psutil.Process()
        rss = process.memory_info().rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
        table.add_row("JOB", Text("{} resident (incl. workers)".format(_binary(rss))))
    except psutil.Error:
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                table.add_row("MPS", Text(_binary(torch.mps.driver_allocated_memory())))
            elif torch.cuda.is_available():
                # Between steps only weights/optimizer state are allocated; per-step
                # activations come and go, so also show the peak and what the process holds.
                _free, total = torch.cuda.mem_get_info()
                reserved = torch.cuda.memory_reserved()
                table.add_row(
                    "CUDA",
                    _gauge(
                        reserved / total,
                        "{} in use · peak {} · {} held of {}".format(
                            _binary(torch.cuda.memory_allocated()),
                            _binary(torch.cuda.max_memory_allocated()),
                            _binary(reserved),
                            _binary(total),
                        ),
                    ),
                )
        except Exception:  # GPU stats are best-effort decoration
            pass
    disk = shutil.disk_usage(tempfile.gettempdir())
    table.add_row("DISK", _gauge(disk.used / disk.total, "{} free".format(_binary(disk.free))))
    return table


def _ticker_bar(items: Sequence[Text], width: int, hints: str = KEY_HINTS) -> Text:
    if not items:
        return Text(" " + hints, style="dim")
    line = Text(" ")
    for item in items:
        line.append_text(item)
        line.append("   ·   ", style="dim")
    if len(line) <= width:
        return line
    doubled = line + line
    offset = int(time.monotonic() * 4) % len(line)
    return doubled.divide([offset, offset + width])[1]


def _compose(view: Mapping[str, Any], width: int, height: int) -> Layout:
    """Lay out one dashboard frame; shared by the live view and the watcher."""
    compact = width < COMPACT_WIDTH or height < COMPACT_HEIGHT
    progress_height = max(3, int(view["progress_rows"]) + 2)
    root = Layout()
    root.split_column(
        Layout(view["header"], size=1),
        Layout(_panel(view["progress"], "PROGRESS", view.get("hints", "")), size=progress_height),
        Layout(name="main"),
        Layout(view["ticker"], size=1),
    )
    board = _panel(view["board"], view["board_title"])
    if compact:
        root["main"].split_column(Layout(board, ratio=1), Layout(name="tape", ratio=1))
        tape_lines = (height - 2 - progress_height) // 2 - 2
    else:
        metrics_height = int(view["metrics_rows"]) + 3 if view["metrics_rows"] else 0
        tape_lines = height - 2 - progress_height - metrics_height - 2
        left = []
        if metrics_height:
            left.append(Layout(_panel(view["metrics"], "METRICS"), size=metrics_height))
        left.append(Layout(name="tape"))
        right = [Layout(board)]
        if view.get("equity") is not None:
            right.append(Layout(_panel(view["equity"], view["equity_title"]), size=view["equity"].height() + 2))
        right.append(Layout(_panel(view["system"], "SYSTEM"), size=8))
        root["main"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
        root["left"].split_column(*left)
        root["right"].split_column(*right)
    body, title = view["tape"](tape_lines)
    root["tape"].update(_panel(body, title))
    return root


def _eta_seconds() -> Optional[float]:
    if _eta_deadline is not None:
        return max(_eta_deadline - time.monotonic(), 0.0)
    for task in list(_progress.tasks):
        if task.fields.get("depth") == 0 and task.total and task.time_remaining is not None:
            return task.time_remaining
    return None


def _live_view(width: int) -> Dict[str, Any]:
    metrics = dict(_metrics)
    entries = list(_tape)
    return {
        "header": _header_bar(
            _title,
            dict(_header),
            time.monotonic() - _started,
            _eta_seconds(),
            "■ STOP AFTER TARGET" if _stop.is_set() else "",
        ),
        "progress": _progress,
        "progress_rows": len(_progress.tasks),
        "hints": KEY_HINTS,
        "metrics": _metrics_table(metrics),
        "metrics_rows": sum(1 for entry in metrics.values() if entry["values"]),
        "tape": lambda lines: _tape_body(entries, lines, _tape_offset, _important_only),
        "board": _board if _board is not None else Text("—", style="dim"),
        "board_title": _board_title,
        "equity": None if _equity is None else _Chart(_equity[1], _equity[2]),
        "equity_title": "" if _equity is None else _equity[0],
        "system": _system_table(),
        "ticker": _ticker_bar(list(dict(_ticker).values()), width),
    }


def _render_live() -> Layout:
    width, height = _console.size
    return _compose(_live_view(width), width, height)


# ------------------------------------------------------- status snapshots


def _snapshot(status: str) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "title": _title,
        "status": status,
        "header": dict(_header),
        "started_at": _started_wall,
        "updated_at": time.time(),
        "elapsed": time.monotonic() - _started,
        "eta": _eta_seconds(),
        "stop_requested": _stop.is_set(),
        "summary": _summary,
        "progress_rows": len(_progress.tasks),
        "progress_ansi": _ansi(_progress, 160),
        "metrics": {
            label: {"values": list(entry["values"]), "higher": entry["higher"]}
            for label, entry in dict(_metrics).items()
        },
        "tape": [[stamp, line.markup, detail] for stamp, line, detail in list(_tape)[-400:]],
        "board_title": _board_title,
        "board_ansi": _ansi(_board, 90) if _board is not None else "",
        "equity": None if _equity is None else {"title": _equity[0], "series": _equity[1], "marks": _equity[2]},
        "system_ansi": _ansi(_system_table(), 70),
        "ticker": [item.markup for item in dict(_ticker).values()],
    }


def _write_status(status: str = "RUNNING") -> None:
    global _status_error_reported
    location = _status_location
    if location is None:
        return
    try:
        document = json.dumps(_snapshot(status), ensure_ascii=False)
        from market_guided_stock_alpha import storage  # lazy: storage imports this module

        if storage.is_remote(location):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "dashboard_status.json"
                path.write_text(document, encoding="utf-8")
                storage.publish(path, location, overwrite=True, quiet=True)
        else:
            target = Path(location)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            partial.write_text(document, encoding="utf-8")
            os.replace(str(partial), str(target))
    except Exception as error:  # status is a convenience; never fail the job
        if not _status_error_reported:
            _status_error_reported = True
            note(("status upload failed ({}); continuing".format(error), "yellow"))


def _status_loop() -> None:
    while not _finished:
        time.sleep(STATUS_INTERVAL_SECONDS)
        if not _finished:
            _check_control()
            _write_status()


def _check_control() -> None:
    """Honour a remote stop request (``MGSA_CONTROL``, set by the job agent)."""
    location = os.environ.get("MGSA_CONTROL")
    if not location or _stop.is_set() or not _stoppable:
        return
    try:
        from market_guided_stock_alpha import storage  # lazy: storage imports this module

        if storage.is_remote(location):
            with tempfile.TemporaryDirectory() as directory:
                text = storage.materialize(location, Path.cwd(), Path(directory), quiet=True).read_text(encoding="utf-8")
        else:
            text = Path(location).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return
    try:
        requested = bool(json.loads(text).get("stop"))
    except ValueError:
        return
    if requested:
        _stop.set()
        note(("■ remote stop requested: finishing the current target, then saving", "bold yellow"))


def snapshot_view(snapshot: Mapping[str, Any], width: int) -> Dict[str, Any]:
    """Rebuild a dashboard view from a status snapshot (for scripts/watch.py)."""
    age = max(time.time() - float(snapshot["updated_at"]), 0.0)
    eta = snapshot.get("eta")
    status = str(snapshot.get("status", "RUNNING"))
    entries = [(stamp, Text.from_markup(markup), bool(detail)) for stamp, markup, detail in snapshot["tape"]]
    fields = dict(snapshot.get("header", {}))
    fields["watch"] = "updated {} ago".format(_duration(age))
    flag = "" if status == "RUNNING" else status
    if snapshot.get("stop_requested") and status == "RUNNING":
        flag = "■ STOP AFTER TARGET"
    equity = snapshot.get("equity")
    return {
        "header": _header_bar(
            str(snapshot["title"]),
            fields,
            float(snapshot["elapsed"]) + (age if status == "RUNNING" else 0.0),
            None if eta is None else max(float(eta) - age, 0.0),
            flag,
        ),
        "progress": Text.from_ansi(snapshot["progress_ansi"]),
        "progress_rows": int(snapshot["progress_rows"]),
        "hints": WATCH_HINTS,
        "metrics": _metrics_table(snapshot.get("metrics", {})),
        "metrics_rows": sum(1 for entry in snapshot.get("metrics", {}).values() if entry["values"]),
        "tape": lambda lines: _tape_body(entries, lines, _tape_offset, _important_only),
        "board": Text.from_ansi(snapshot["board_ansi"]) if snapshot["board_ansi"] else Text("—", style="dim"),
        "board_title": snapshot["board_title"],
        "equity": None if equity is None else _Chart([tuple(s) for s in equity["series"]], equity["marks"]),
        "equity_title": "" if equity is None else equity["title"],
        "system": Text.from_ansi(snapshot["system_ansi"]),
        "ticker": _ticker_bar([Text.from_markup(item) for item in snapshot.get("ticker", [])], width, WATCH_HINTS),
    }


def watch_keys(enabled: bool) -> None:
    """Enable (or restore) tape keys for the watcher's own full-screen view."""
    if enabled:
        _start_keys()
    else:
        _stop_keys()


def compose(view: Mapping[str, Any], width: int, height: int) -> Layout:
    return _compose(view, width, height)


# --------------------------------------------------------------- public API


def _append_tape(line: Text, detail: bool) -> None:
    stamp = _clock()
    _tape.append((stamp, line, detail))
    _log_line(stamp, line)


def _log_line(stamp: str, line: Text) -> None:
    if _log_file is not None:
        try:
            _log_file.write("{} {}\n".format(stamp, line.plain))
        except ValueError:  # closed during shutdown
            pass


class _Bar(object):
    """The tqdm-like surface the codebase uses: iterate, update, postfix, close."""

    def __init__(
        self,
        iterable: Optional[Iterable[Any]],
        total: Optional[float],
        desc: str,
        unit: str,
        leave: bool,
    ) -> None:
        _begin()
        self.iterable = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)  # type: ignore[arg-type]
        self.desc = desc
        self.unit = unit
        self.leave = leave
        depth = len(_progress.tasks)
        label = Text("  " * depth + ("↳ " if depth else ""))
        label.append(desc, style=_DEPTH_STYLES[min(depth, len(_DEPTH_STYLES) - 1)])
        self.task: Any = _progress.add_task(label, total=total, unit=unit, postfix="", depth=depth)

    def __iter__(self) -> Iterator[Any]:
        try:
            for item in () if self.iterable is None else self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self) -> "_Bar":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def update(self, n: float = 1) -> None:
        if self.task is not None:
            _progress.advance(self.task, n)

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        if self.task is not None:
            _progress.update(self.task, postfix=text)

    def set_postfix(self, refresh: bool = True, **values: Any) -> None:
        self.set_postfix_str("  ".join("{} {}".format(k, v) for k, v in values.items()))

    def close(self) -> None:
        if self.task is None:
            return
        task = next(t for t in _progress.tasks if t.id == self.task)
        _progress.remove_task(self.task)
        self.task = None
        if self.leave:
            done = (
                filesize.decimal(int(task.completed))
                if self.unit == "B"
                else "{:,} {}".format(int(task.completed), self.unit)
            )
            speed = task.finished_speed or task.speed
            note(
                ("✓ ", "bold green"),
                (self.desc, "bold"),
                ("  {}  in {}".format(done, _duration(task.elapsed or 0)), "dim"),
                ("  {}".format(_rate(speed, self.unit)) if speed else "", "cyan"),
            )


def progress(
    iterable: Optional[Iterable[Any]] = None,
    total: Optional[float] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
) -> _Bar:
    """Progress bar over ``iterable`` (or manual ``update`` against ``total``).

    ``unit="B"`` renders byte sizes and transfer speed. Every bar is removed
    from the PROGRESS panel when it closes; ``leave=True`` also logs a
    completion line to the tape (use ``leave=False`` for inner loops).
    """
    return _Bar(iterable, total, desc, unit, leave)


def note(*parts: Any, detail: bool = False) -> None:
    """Log a line to the tape (and log file); plain stderr line off-terminal.

    Each part is plain text or a ``(text, style)`` pair. ``detail=True`` marks
    routine lines (e.g. per epoch) that the ``t`` filter hides. A single rich
    renderable (e.g. a ``Table``) is printed off-terminal and, on the
    dashboard, kept for the exit summary; the log file gets its plain text.
    """
    live = _begin()
    if len(parts) == 1 and not isinstance(parts[0], (str, tuple)):
        if _log_file is not None:
            _log_file.write(_plain(parts[0]))
        if live:
            _exit_prints.append(parts[0])
        else:
            _console.print(parts[0])
        return
    line = Text()
    for part in parts:
        if isinstance(part, tuple):
            line.append(part[0], style=part[1])
        else:
            line.append(part)
    _append_tape(line, detail)
    if not live:
        _console.print(line)


def board(renderable: Any, title: str = "BOARD") -> None:
    """Show ``renderable`` in the dashboard's main panel."""
    global _board, _board_title
    _begin()
    _board, _board_title = renderable, title


def metric(label: str, value: float, higher_is_better: bool = True) -> None:
    """Append ``value`` to a METRICS row."""
    entry = _metrics.setdefault(label, {"values": [], "higher": higher_is_better})
    entry["values"].append(float(value))


def reset_metrics() -> None:
    _metrics.clear()


def curves(title: str, series: Sequence[Tuple[str, Sequence[float]]], marks: Sequence[int] = ()) -> None:
    """Show braille line charts (e.g. cumulative IC) in the EQUITY panel.

    ``marks`` are x positions drawn as dotted separators (target boundaries).
    """
    global _equity
    _equity = (title, [(label, [float(v) for v in values]) for label, values in series], [int(m) for m in marks])


def ticker(key: str, *parts: Tuple[str, str]) -> None:
    """Set one scrolling ticker item."""
    _ticker[key] = Text.assemble(*parts)


def header(key: str, value: Any) -> None:
    """Show ``key value`` in the header bar."""
    _begin()
    _header[key] = str(value)


def summary(text: str) -> None:
    """One-line result used by the exit notification (e.g. overall rank IC)."""
    global _summary
    _summary = text


def eta(seconds: Optional[float]) -> None:
    """Override the header ETA (seconds from now); ``None`` restores the bar estimate."""
    global _eta_deadline
    _eta_deadline = None if seconds is None else time.monotonic() + max(float(seconds), 0.0)


def allow_stop(enabled: bool = True) -> None:
    """Declare that this job checks ``stop_requested`` at safe points."""
    global _stoppable
    _stoppable = enabled


def stop_requested() -> bool:
    return _stop.is_set()


def log_to(path: Any) -> None:
    """Append every tape line (including earlier ones) to ``path``.

    Active only inside ``run_main`` jobs, so library and test calls leave no
    files; ``DASHBOARD_LOG`` takes precedence when set.
    """
    if _job_mode and not os.environ.get("DASHBOARD_LOG"):
        _open_log(path)


def _open_log(path: Any) -> None:
    global _log_file
    _begin()
    target = Path(str(path))
    if _log_file is not None and _log_file.name == str(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(str(target), "a", encoding="utf-8", buffering=1)
    handle.write("==== {} · {} ====\n".format(_title, datetime.now().isoformat(timespec="seconds")))
    for stamp, line, _detail in list(_tape):
        handle.write("{} {}\n".format(stamp, line.plain))
    if _log_file is not None:
        _log_file.close()
    _log_file = handle


def status_to(location: Any) -> None:
    """Write a JSON status snapshot to ``location`` every 2 s and at exit.

    Active only inside ``run_main`` jobs; ``DASHBOARD_STATUS`` takes precedence.
    """
    if _job_mode and not os.environ.get("DASHBOARD_STATUS"):
        _start_status(location)


def _start_status(location: Any) -> None:
    global _status_location, _status_thread
    _begin()
    _status_location = str(location)
    if _status_thread is None:
        _status_thread = threading.Thread(target=_status_loop, daemon=True)
        _status_thread.start()
    threading.Thread(target=_write_status, daemon=True).start()


def kv_table(mapping: Mapping[str, Any], title: Optional[str] = None) -> Table:
    """Two-column key/value table for summary boards (numbers right-aligned)."""
    table = Table(
        title=title,
        title_style="bold " + ACCENT,
        box=box.SIMPLE_HEAD,
        show_header=False,
        border_style=BORDER,
        padding=(0, 1),
    )
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    for key, value in mapping.items():
        if isinstance(value, Text):
            cell = value
        elif isinstance(value, float):
            cell = Text("{:,.4f}".format(value))
        elif isinstance(value, int) and not isinstance(value, bool):
            cell = Text("{:,}".format(value))
        else:
            cell = Text(str(value))
        table.add_row(str(key), cell)
    return table
