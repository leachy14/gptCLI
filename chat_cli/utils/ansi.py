"""Colour and styling helpers built on :mod:`rich`."""

import os
from rich.console import Console


console = Console()


class Ansi:
    """Lightweight collection of style names used throughout the app."""

    BOLD = "bold"

    FG_GREEN = "green"
    FG_CYAN = "cyan"
    FG_MAGENTA = "magenta"
    FG_YELLOW = "yellow"
    FG_RED = "red"

    @staticmethod
    def style(text: str, *codes: str) -> str:
        """Return *text* wrapped in rich markup unless ``NO_COLOR`` is set."""
        if os.getenv("NO_COLOR") is not None:
            return text
        style = " ".join(codes)
        return f"[{style}]{text}[/]"


# Common labels used throughout the application
USER_LABEL = Ansi.style("you", Ansi.FG_CYAN, Ansi.BOLD)
ASSISTANT_LABEL = Ansi.style("assistant", Ansi.FG_GREEN, Ansi.BOLD)
ERROR_LABEL = Ansi.style("error", Ansi.FG_RED, Ansi.BOLD)
WARNING_LABEL = Ansi.style("warning", Ansi.FG_YELLOW, Ansi.BOLD)
REASONING_LABEL = Ansi.style("reasoning", Ansi.FG_MAGENTA, Ansi.BOLD) 