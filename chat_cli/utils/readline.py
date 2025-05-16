"""Readline utilities for improved terminal input handling."""

import re


# Regex that matches ANSI CSI escape sequences (e.g. "\033[92m"). It is
# intentionally simple because we only need to wrap, not validate.
_ANSI_PATTERN = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def readline_safe_prompt(prompt: str) -> str:
    """Return *prompt* with ANSI escapes wrapped for correct Readline width.
    
    When coloured ANSI escape sequences are included in the prompt string that
    is passed to `input()`, GNU Readline (used by Python for interactive input
    if available) counts those bytes as **printable** characters unless
    instructed otherwise. This results in mis-aligned cursor positioning and
    broken line wrapping once the user types beyond the terminal width.
    
    Readline solves this by allowing non-printing parts to be wrapped between
    the control characters \001 (start of *hidden* sequence) and \002 (end).
    """
    if "\033[" not in prompt:  # fast-path – no colour codes present
        return prompt

    # Insert \001/\002 around each escape sequence. We must keep the escape
    # codes themselves unchanged so the terminal still interprets them.
    return _ANSI_PATTERN.sub(lambda m: f"\001{m.group(0)}\002", prompt) 