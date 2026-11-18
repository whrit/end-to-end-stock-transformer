"""Terminal progress bars for long-running loops (tqdm).

Bars write to stderr and switch themselves off when stderr is not a terminal
(pytest, pipes, log files). Standard tqdm environment variables apply, e.g.
``TQDM_DISABLE=1`` hides every bar and ``TQDM_MININTERVAL=5`` refreshes less often.
"""

import os
import sys

from tqdm.auto import tqdm


def progress(iterable: Optional[Iterable[Any]] = None, **kwargs: Any) -> Any:
    """Return a tqdm bar; auto-disabled off-terminal unless TQDM_DISABLE is set."""
    if "TQDM_DISABLE" not in os.environ:
        kwargs.setdefault("disable", None)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, **kwargs)


def note(message: str) -> None:
    """Print a stage message to stderr without breaking an active progress bar.

    stderr keeps stdout clean for the JSON manifests scripts print.
    """
    tqdm.write(message, file=sys.stderr)
