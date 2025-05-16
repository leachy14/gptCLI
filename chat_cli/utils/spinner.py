"""Terminal spinner animation for indicating progress."""

import itertools
import sys
import threading
import time
from typing import Optional


class Spinner:
    """A minimal terminal spinner shown while waiting for the first token.

    Usage:
        spinner = Spinner(prefix="assistant> ")
        spinner.start()
        # do blocking work …
        spinner.stop()  # ensures the line is reset to `prefix` again
    """

    _cycle = itertools.cycle("|/-\\")

    def __init__(self, prefix: str = "", delay: float = 0.1):
        self._prefix = prefix
        self._delay = delay
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _spin(self) -> None:  # runs in a background thread
        while not self._stop_event.is_set():
            ch = next(self._cycle)
            # Carriage return to beginning-of-line, then redraw prefix + frame
            sys.stdout.write(f"\r{self._prefix}{ch}")
            sys.stdout.flush()
            time.sleep(self._delay)

    def start(self) -> None:
        if self._thread is not None:
            return  # already running
        # Print prefix once so we always redraw at same length
        sys.stdout.write(self._prefix)
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        # Clear spinner character and redraw prefix ready for real text
        sys.stdout.write(f"\r{self._prefix}")
        sys.stdout.flush() 