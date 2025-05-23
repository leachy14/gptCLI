"""Spinner utility using yaspin with fallback to a simple implementation."""
from __future__ import annotations

import sys
import itertools
import threading
import time
from typing import Optional

from .ansi import console

try:
    from yaspin import yaspin  # type: ignore
except Exception:  # pragma: no cover - optional dep may be missing
    yaspin = None  # type: ignore


class Spinner:
    """Display a small spinner next to a prefix while work is done."""

    def __init__(self, prefix: str = "", delay: float = 0.1):
        self._prefix = prefix
        self._delay = delay
        self._started = False
        if yaspin is not None:
            # spinner after the text so prefix stays at the start
            self._spinner = yaspin(text="", side="right")
        else:
            self._cycle = itertools.cycle("|/-\\")
            self._stop_event = threading.Event()
            self._thread: Optional[threading.Thread] = None

    def _spin(self) -> None:
        while not self._stop_event.is_set():
            ch = next(self._cycle)
            console.print(f"\r{self._prefix}{ch}", end="")
            console.file.flush()
            time.sleep(self._delay)

    def start(self) -> None:
        if self._started:
            return
        if yaspin is not None:
            console.print(self._prefix, end="")
            console.file.flush()
            self._spinner.start()
        else:
            console.print(self._prefix, end="")
            console.file.flush()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        if yaspin is not None:
            self._spinner.stop()
            console.print(f"\r{self._prefix}", end="")
            console.file.flush()
        else:
            if self._thread:
                self._stop_event.set()
                self._thread.join()
                self._thread = None
                self._stop_event.clear()
            console.print(f"\r{self._prefix}", end="")
            console.file.flush()
        self._started = False
