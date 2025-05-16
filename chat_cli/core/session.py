"""Session management for chat conversations."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..utils.ansi import Ansi

# A single, persistent system message ensures the model is aware that it is
# interacting in a terminal context and should optimise readability for that
# form factor.
SYSTEM_PROMPT = (
    "You are an AI assistant running in a terminal (CLI) environment. "
    "Optimise all answers for 80‑column readability, prefer plain text, "
    "ASCII art or concise bullet lists over heavy markup, and wrap code "
    "snippets in fenced blocks when helpful. Do not emit trailing spaces or "
    "control characters."
)

# Supported models
SUPPORTED_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",  # default
    "o1",
    "o3",
    "o4-mini",
]


class Session:
    """Represents a chat session stored as a JSON file on disk."""

    FILENAME_SUFFIX = ".json"
    SESSIONS_DIR = Path.home() / ".chat_cli_sessions"

    def __init__(
        self,
        name: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        enable_web_search: bool = True,
        enable_reasoning_summary: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        self.messages: List[Dict[str, Any]] = messages or []
        self.enable_web_search = enable_web_search
        self.enable_reasoning_summary = enable_reasoning_summary

        # Ensure the very first message in every session is our fixed system
        # prompt so the model formats its output appropriately.  We insert it
        # only once to avoid polluting the history when a session is loaded
        # from disk multiple times.
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # Create sessions directory if it doesn't exist
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.SESSIONS_DIR / f"{self.name}{self.FILENAME_SUFFIX}"

    def save(self) -> None:
        data = {
            "model": self.model,
            "messages": self.messages,
            "enable_web_search": self.enable_web_search,
            "enable_reasoning_summary": self.enable_reasoning_summary,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp_path = self.path.with_suffix(".tmp")
        # If any unexpected non‑serializable objects slip through we coerce
        # them to strings to avoid breaking the CLI's persistence mechanism.
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )
        tmp_path.rename(self.path)

    @classmethod
    def load(cls, name: str) -> "Session":
        path = cls.SESSIONS_DIR / f"{name}{cls.FILENAME_SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"Session '{name}' does not exist.")
        data = json.loads(path.read_text())
        return cls(
            name=name,
            model=data.get("model", "gpt-4o-mini"),
            messages=data.get("messages", []),
            enable_web_search=data.get("enable_web_search", False),
            enable_reasoning_summary=data.get("enable_reasoning_summary", False),
        )

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    # For function & tool call raw objects we might want to store them as‑is
    def add_raw(self, item: Dict[str, Any]) -> None:
        self.messages.append(item)

    @classmethod
    def list_sessions(cls) -> None:
        files = sorted(cls.SESSIONS_DIR.glob(f"*{cls.FILENAME_SUFFIX}"))
        if not files:
            print("(no saved sessions)")
            return
        print(Ansi.style("Saved sessions:", Ansi.BOLD, Ansi.FG_MAGENTA))
        for file in files:
            name = file.stem
            updated = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            indicator_char = "★" if file == (cls.SESSIONS_DIR / f"{name}{cls.FILENAME_SUFFIX}") else " "
            colour = Ansi.FG_GREEN if indicator_char == "★" else Ansi.FG_CYAN
            label = Ansi.style(name, colour)
            print(f"  {indicator_char} {label} (updated: {updated})") 