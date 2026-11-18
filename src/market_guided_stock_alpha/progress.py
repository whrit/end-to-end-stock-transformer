"""Trading-terminal style live dashboard for every long-running job (rich).

The first bar or note in an interactive terminal opens one full-screen
dashboard for the rest of the process:

    header   job title, extra fields, clock, elapsed, ETA of the top-level bar
    PROGRESS every open bar, nested bars indented under their parent
    METRICS  latest values with ▲/▼ change, best, and a sparkline (training)
    TAPE     timestamped log lines (``note``), warnings, stray stderr
    BOARD    the job's main table (fetch stages, rolling target blotter)
    SYSTEM   CPU, RAM (this job incl. worker processes), GPU memory, temp disk
    ticker   scrolling headline results

When the process ends (normally, by exception or Ctrl-C) the screen closes and
a summary (board, saved tables, recent tape) is printed into the scrollback.
Stdout written meanwhile (JSON manifests) is replayed afterwards.

Off-terminal (pytest, pipes, log files) there is no dashboard: notes print as
plain lines, bars and board/metric/ticker updates are skipped.
``PROGRESS_DISABLE=1`` forces that plain mode; ``NO_COLOR=1`` drops colour.
"""

import atexit
import os
import shutil
import sys
import tempfile
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import psutil
from rich import box, filesize
from rich.console import Console, Group
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
from rich.table import Column, Table
from rich.text import Text

ACCENT = "dodger_blue1"
ACCENT_BAR = "bold white on dodger_blue3"
BORDER = "grey37"
SPARK = "▁▂▃▄▅▆▇█"
_DEPTH_STYLES = ("bold cyan", "magenta", "blue", "bright_black")

_console = Console(stderr=True, highlight=False)


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

# Dashboard state. Mutated from the job thread, read by the refresh thread;
# readers copy containers first (list()/dict() are atomic under the GIL).
_live: Optional[Live] = None
_started = 0.0
_title = ""
_header: Dict[str, str] = {}
_tape: Deque[Tuple[str, Text]] = deque(maxlen=500)
_metrics: Dict[str, Dict[str, Any]] = {}
_board: Any = None
_board_title = "BOARD"
_ticker: Dict[str, Text] = {}
_exit_prints: List[Any] = []
_stdout_buffer: List[str] = []
_saved: Dict[str, Any] = {}
_atexit_registered = False


def _enabled() -> bool:
    return os.environ.get("PROGRESS_DISABLE", "") in ("", "0") and _console.is_terminal


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
    """Stray Python-level stderr writes become dim tape lines."""

    encoding = "utf-8"

    def __init__(self) -> None:
        self.partial = ""

    def write(self, text: str) -> int:
        lines = (self.partial + text).split("\n")
        self.partial = lines.pop()
        for line in lines:
            if line.strip():
                _tape.append((_clock(), Text(line.rstrip(), style="dim")))
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _start() -> bool:
    """Open the dashboard if possible; True when it is live."""
    global _live, _started, _title, _atexit_registered
    if _live is not None:
        return True
    if not _enabled():
        return False
    _started = time.monotonic()
    _title = " ".join(
        [os.path.basename(sys.argv[0] or "python")] + [os.path.basename(a) for a in sys.argv[1:3]]
    )
    psutil.cpu_percent(None)
    _console.file = sys.stderr  # pin before stderr is redirected into the tape
    _saved.update(
        showwarning=warnings.showwarning,
        excepthook=sys.excepthook,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    warnings.showwarning = _show_warning
    sys.excepthook = _excepthook
    if sys.stdout.isatty():
        sys.stdout = _StdoutBuffer()  # type: ignore[assignment]
    sys.stderr = _StderrToTape()  # type: ignore[assignment]
    _live = Live(
        console=_console,
        screen=True,
        refresh_per_second=4,
        redirect_stdout=False,
        redirect_stderr=False,
        get_renderable=_render,
    )
    _live.start()
    if not _atexit_registered:
        atexit.register(_finish, "DONE")
        _atexit_registered = True
    return True


def _finish(status: str) -> None:
    """Close the full-screen view and print the summary into the scrollback."""
    global _live
    if _live is None:
        return
    _live.stop()
    _live = None
    warnings.showwarning = _saved["showwarning"]
    sys.excepthook = _saved["excepthook"]
    sys.stdout = _saved["stdout"]
    sys.stderr = _saved["stderr"]
    elapsed = _duration(time.monotonic() - _started)
    style = {"DONE": "bold green", "FAILED": "bold red"}.get(status, "bold yellow")
    _console.print(
        Rule(
            Text.assemble(
                ("MGSA ▸ {} ".format(_title), "bold " + ACCENT),
                (status, style),
                (" · elapsed {}".format(elapsed), "dim"),
            ),
            style=ACCENT,
        )
    )
    if _exit_prints:
        for renderable in _exit_prints:
            _console.print(renderable)
    elif _board is not None:
        _console.print(_board)
    recent = list(_tape)[-12:]
    if recent:
        _console.print(Text("recent tape", style="bold " + ACCENT))
        for stamp, line in recent:
            _console.print(Text.assemble((stamp + " ", "dim"), line))
    _console.file = None  # type: ignore[assignment]
    if _stdout_buffer:
        sys.stdout.write("".join(_stdout_buffer))
        sys.stdout.flush()
        del _stdout_buffer[:]


def _excepthook(kind: Any, value: Any, traceback: Any) -> None:
    hook = _saved["excepthook"]
    _finish("INTERRUPTED" if issubclass(kind, KeyboardInterrupt) else "FAILED")
    hook(kind, value, traceback)


def _show_warning(message: Any, category: Any, filename: str, lineno: int, *_: Any) -> None:
    note(
        ("⚠ {}: ".format(category.__name__), "bold yellow"),
        (str(message).strip(), "yellow"),
        ("  ({}:{})".format(os.path.basename(filename), lineno), "dim"),
    )


# ---------------------------------------------------------------- rendering


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _panel(content: Any, title: str) -> Panel:
    return Panel(
        content,
        title=Text(" {} ".format(title), style=ACCENT_BAR),
        title_align="left",
        border_style=BORDER,
        box=box.SQUARE,
    )


def _header_bar() -> Table:
    eta = ""
    for task in list(_progress.tasks):
        if task.fields.get("depth") == 0 and task.total and task.time_remaining is not None:
            eta = "  ETA {}".format(_duration(task.time_remaining))
            break
    left = Text(" MGSA ▸ {}".format(_title), style="bold")
    for key, value in dict(_header).items():
        left.append("  │  {} {}".format(key, value))
    right = Text(
        "{}  │  elapsed {}{} ".format(_clock(), _duration(time.monotonic() - _started), eta)
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


def _metrics_panel() -> Panel:
    table = Table(box=None, expand=True, padding=(0, 1), header_style="dim")
    for name, justify in (("metric", "left"), ("last", "right"), ("chg", "right"), ("best", "right"), ("trend", "left")):
        table.add_column(name, justify=justify, no_wrap=True)
    for label, entry in dict(_metrics).items():
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
    return _panel(table, "METRICS")


def _tape_panel(lines: int) -> Panel:
    rows = []
    for stamp, line in list(_tape)[-max(lines, 1):]:
        row = Text.assemble((stamp + " ", "dim"), line)
        row.no_wrap = True
        row.overflow = "ellipsis"
        rows.append(row)
    return _panel(Group(*rows), "TAPE")


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


def _system_panel() -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    cpu = psutil.cpu_percent(None)
    table.add_row("CPU", _gauge(cpu / 100.0, "{:.0f}%".format(cpu)))
    memory = psutil.virtual_memory()
    table.add_row(
        "RAM",
        _gauge(
            memory.percent / 100.0,
            "{}/{}".format(_binary(memory.used), _binary(memory.total)),
        ),
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
                table.add_row("CUDA", Text(_binary(torch.cuda.memory_allocated())))
        except Exception:  # GPU stats are best-effort decoration
            pass
    disk = shutil.disk_usage(tempfile.gettempdir())
    table.add_row(
        "DISK", _gauge(disk.used / disk.total, "{} free".format(_binary(disk.free)))
    )
    return _panel(table, "SYSTEM")


def _ticker_bar(width: int) -> Text:
    items = list(dict(_ticker).values())
    if not items:
        return Text(" Ctrl-C stops the job · PROGRESS_DISABLE=1 for plain logs", style="dim")
    line = Text(" ")
    for item in items:
        line.append_text(item)
        line.append("   ·   ", style="dim")
    if len(line) <= width:
        return line
    doubled = line + line
    offset = int(time.monotonic() * 4) % len(line)
    return doubled.divide([offset, offset + width])[1]


def _render() -> Layout:
    width, height = _console.size
    progress_height = max(3, len(_progress.tasks) + 2)
    metrics_height = (len(_metrics) + 3) if _metrics else 0
    tape_height = max(3, height - 2 - progress_height - metrics_height)
    left = []
    if metrics_height:
        left.append(Layout(_metrics_panel(), size=metrics_height))
    left.append(Layout(_tape_panel(tape_height - 2)))
    board = _board if _board is not None else Text("—", style="dim")
    root = Layout()
    root.split_column(
        Layout(_header_bar(), size=1),
        Layout(_panel(_progress, "PROGRESS"), size=progress_height),
        Layout(name="main"),
        Layout(_ticker_bar(width), size=1),
    )
    root["main"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
    root["left"].split_column(*left)
    root["right"].split_column(
        Layout(_panel(board, _board_title)), Layout(_system_panel(), size=8)
    )
    return root


# --------------------------------------------------------------- public API


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
        self.iterable = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)  # type: ignore[arg-type]
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self.task: Any = None
        if _start():
            depth = len(_progress.tasks)
            label = Text("  " * depth + ("↳ " if depth else ""))
            label.append(desc, style=_DEPTH_STYLES[min(depth, len(_DEPTH_STYLES) - 1)])
            self.task = _progress.add_task(label, total=total, unit=unit, postfix="", depth=depth)

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


def note(*parts: Any) -> None:
    """Log a line to the tape (plain stderr line off-terminal).

    Each part is plain text or a ``(text, style)`` pair. A single rich
    renderable (e.g. a ``Table``) is printed off-terminal and, on the
    dashboard, kept for the exit summary.
    """
    live = _start()
    if len(parts) == 1 and not isinstance(parts[0], (str, tuple)):
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
    if live:
        _tape.append((_clock(), line))
    else:
        _console.print(line)


def board(renderable: Any, title: str = "BOARD") -> None:
    """Show ``renderable`` in the dashboard's main panel (no-op off-terminal)."""
    global _board, _board_title
    if _start():
        _board, _board_title = renderable, title


def metric(label: str, value: float, higher_is_better: bool = True) -> None:
    """Append ``value`` to a METRICS row (no-op off-terminal)."""
    if _live is None:
        return
    entry = _metrics.setdefault(label, {"values": [], "higher": higher_is_better})
    entry["values"].append(float(value))


def reset_metrics() -> None:
    _metrics.clear()


def ticker(key: str, *parts: Tuple[str, str]) -> None:
    """Set one scrolling ticker item (no-op off-terminal)."""
    if _live is not None:
        _ticker[key] = Text.assemble(*parts)


def header(key: str, value: Any) -> None:
    """Show ``key value`` in the header bar (no-op off-terminal)."""
    if _start():
        _header[key] = str(value)
