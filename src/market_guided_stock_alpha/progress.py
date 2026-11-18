"""Colourful terminal progress for long-running loops (rich).

All bars share one live display on stderr; nested bars stack under their parent
with indentation. ``note`` prints styled log lines above the bars. Python
warnings raised while bars are live print as one tidy line instead of breaking
the display.

Bars only render on an interactive terminal; off-terminal (pytest, pipes, log
files) they are skipped and notes print as plain text. ``PROGRESS_DISABLE=1``
hides bars everywhere; ``NO_COLOR=1`` drops colour.
"""

import os
import warnings
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

from rich import filesize
from rich.console import Console
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
from rich.table import Column
from rich.text import Text

Part = Union[str, Tuple[str, str]]

_console = Console(stderr=True, highlight=False)
_DEPTH_STYLES = ("bold cyan", "magenta", "blue", "bright_black")
_live: Optional[Progress] = None
_open_bars = 0
_saved_showwarning: Any = None


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
            return Text("", style="cyan")
        unit = task.fields.get("unit", "it")
        if unit == "B":
            return Text("{}/s".format(filesize.decimal(int(speed))), style="cyan")
        if speed >= 1:
            return Text("{:.1f} {}/s".format(speed, unit), style="cyan")
        return Text("{:.1f} s/{}".format(1.0 / speed, unit), style="cyan")


def _display() -> Progress:
    global _live, _saved_showwarning
    if _live is None:
        _live = Progress(
            SpinnerColumn("dots", style="cyan", finished_text=Text("✓", style="bold green")),
            TextColumn(
                "{task.description}",
                table_column=Column(width=40, no_wrap=True, overflow="ellipsis"),
            ),
            BarColumn(
                bar_width=32,
                style="grey35",
                complete_style="magenta",
                finished_style="green",
                pulse_style="cyan",
            ),
            TaskProgressColumn(),
            _CountColumn(),
            _RateColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            TextColumn(
                "{task.fields[postfix]}",
                style="yellow",
                markup=False,
                table_column=Column(no_wrap=True, overflow="ellipsis"),
            ),
            console=_console,
            refresh_per_second=8,
        )
        _live.start()
        _saved_showwarning = warnings.showwarning
        warnings.showwarning = _show_warning
    return _live


def _release() -> None:
    global _live, _open_bars
    _open_bars -= 1
    if _open_bars == 0 and _live is not None:
        _live.stop()
        _live = None
        warnings.showwarning = _saved_showwarning


def _show_warning(message: Any, category: Any, filename: str, lineno: int, *_: Any) -> None:
    note(
        ("⚠ {}: ".format(category.__name__), "bold yellow"),
        (str(message).strip(), "yellow"),
        ("  ({}:{})".format(os.path.basename(filename), lineno), "dim"),
    )


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
        global _open_bars
        self.iterable = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)  # type: ignore[arg-type]
        self.leave = leave
        self.task: Any = None
        self.display: Optional[Progress] = None
        if os.environ.get("PROGRESS_DISABLE", "") in ("", "0") and _console.is_terminal:
            depth = _open_bars
            _open_bars += 1
            self.display = _display()
            label = Text("  " * depth + ("↳ " if depth else ""))
            label.append(desc, style=_DEPTH_STYLES[min(depth, len(_DEPTH_STYLES) - 1)])
            self.task = self.display.add_task(label, total=total, unit=unit, postfix="")

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
        if self.display is not None:
            self.display.advance(self.task, n)

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        if self.display is not None:
            self.display.update(self.task, postfix=text)

    def set_postfix(self, refresh: bool = True, **values: Any) -> None:
        self.set_postfix_str("  ".join("{} {}".format(k, v) for k, v in values.items()))

    def close(self) -> None:
        if self.display is None:
            return
        if self.leave:
            task = self.display.tasks[[t.id for t in self.display.tasks].index(self.task)]
            if task.total is None:
                self.display.update(self.task, total=task.completed)
        else:
            self.display.remove_task(self.task)
        self.display = None
        _release()


def progress(
    iterable: Optional[Iterable[Any]] = None,
    total: Optional[float] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
) -> _Bar:
    """Progress bar over ``iterable`` (or manual ``update`` against ``total``).

    ``unit="B"`` renders byte sizes and transfer speed. ``leave=False`` removes
    the bar when it finishes (use for inner loops).
    """
    return _Bar(iterable, total, desc, unit, leave)


def note(*parts: Any) -> None:
    """Print a log line above any live bars.

    Each part is plain text, a ``(text, style)`` pair, or a rich renderable
    (e.g. a ``Table``). Styles drop automatically off-terminal.
    """
    if len(parts) == 1 and not isinstance(parts[0], (str, tuple)):
        _console.print(parts[0])
        return
    line = Text()
    for part in parts:
        if isinstance(part, tuple):
            line.append(part[0], style=part[1])
        else:
            line.append(part)
    _console.print(line)
