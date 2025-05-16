"""ANSI color and styling utilities for terminal output."""

import os


class Ansi:
    """Lightweight collection of ANSI escape codes we rely on."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    FG_GREEN = "\033[92m"
    FG_CYAN = "\033[96m"
    FG_MAGENTA = "\033[95m"
    FG_YELLOW = "\033[93m"
    FG_RED = "\033[91m"

    @staticmethod
    def style(text: str, *codes: str) -> str:
        """Return `text` wrapped in the given ANSI codes unless NO_COLOR set."""
        if os.getenv("NO_COLOR") is not None:
            return text
        return "".join(codes) + text + Ansi.RESET


# Common labels used throughout the application
USER_LABEL = Ansi.style("you", Ansi.FG_CYAN, Ansi.BOLD)
ASSISTANT_LABEL = Ansi.style("assistant", Ansi.FG_GREEN, Ansi.BOLD)
ERROR_LABEL = Ansi.style("error", Ansi.FG_RED, Ansi.BOLD)
WARNING_LABEL = Ansi.style("warning", Ansi.FG_YELLOW, Ansi.BOLD)
REASONING_LABEL = Ansi.style("reasoning", Ansi.FG_MAGENTA, Ansi.BOLD) 