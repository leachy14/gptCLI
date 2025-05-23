"""Terminal Chat CLI built on top of OpenAI models.

This is an extracted version of the original single-file implementation,
refactored into a small package for improved maintainability.
"""
from __future__ import annotations

import argparse
import os
import sys
import readline  # noqa: F401 – side-effect: history & line editing
from rich.panel import Panel
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import questionary  # type: ignore
except Exception:  # pragma: no cover - library might be missing in tests
    class _DummyQuestionary:
        def select(self, *args, **kwargs):
            raise RuntimeError("questionary library is required")

    questionary = _DummyQuestionary()

from openai import OpenAI  # type: ignore

from .core import Session, SUPPORTED_MODELS, SYSTEM_PROMPT
from .core.client import OpenAIClientWrapper
from .utils import (
    Ansi,
    USER_LABEL,
    console,
)

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class ChatCLI:
    """High-level orchestration class for the interactive REPL."""

    def __init__(self, initial_session: Session, client_wrapper: OpenAIClientWrapper):
        self.session = initial_session
        self.client = client_wrapper

    # ---------------- Utility ----------------

    def list_sessions(self) -> None:
        Session.list_sessions(current=self.session.name)

    # -------------- Interactive pickers ---------------

    @staticmethod
    def _interactive_picker(
        title: str, options: List[str], current: Optional[str] = None
    ) -> Optional[str]:
        """Present *options* to the user and return the selected value."""
        if not options:
            console.print("(no items available)")
            return None

        if questionary is not None:
            try:
                return questionary.select(
                    title,
                    choices=options,
                    default=current,
                ).ask()
            except (KeyboardInterrupt, EOFError):
                print()
                return None

        # Fallback if questionary is unavailable
        print(Ansi.style(title, Ansi.BOLD, Ansi.FG_MAGENTA))

        for idx, item in enumerate(options, start=1):
            star = "\u2190 current" if current and item == current else ""
            colour = Ansi.FG_GREEN if star else Ansi.FG_CYAN
            console.print(f"  {idx}. {Ansi.style(item, colour)} {star}")

        try:
            choice_str = console.input("Select number (Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None

        if not choice_str:
            return None  # cancelled

        if not choice_str.isdigit():
            console.print(Ansi.style("Invalid selection – expected a number.", Ansi.FG_RED))
            return None

        choice = int(choice_str)
        if not (1 <= choice <= len(options)):
            console.print(Ansi.style("Selection out of range.", Ansi.FG_RED))
            return None

        return options[choice - 1]

    # ---------------- Command handling ---------------

    def handle_command(self, line: str) -> bool:
        """Handle slash commands. Return False to exit REPL."""

        parts = line.strip().split()
        if not parts:
            return True

        cmd = parts[0].lower()

        if cmd == "/help":
            from . import __doc__ as _doc  # lazy import to avoid circularity

            console.print(_doc or "(no help available)")

        elif cmd == "/exit":
            self.session.save()
            console.print("Session saved. Bye!")
            return False

        elif cmd == "/model":
            if len(parts) == 1:
                selection = self._interactive_picker(
                    "Select a model:", SUPPORTED_MODELS, current=self.session.model
                )
                if selection and selection in SUPPORTED_MODELS:
                    self.session.model = selection
                    console.print(f"[model switched to {self.session.model}]")
                return True

            if len(parts) != 2:
                console.print("Usage: /model <model_name>")
            else:
                model_name = parts[1]
                if model_name not in SUPPORTED_MODELS:
                    console.print("Unsupported model. Use /models to see the list of supported models.")
                else:
                    self.session.model = model_name
                    console.print(f"[model switched to {self.session.model}]")

        elif cmd == "/models":
            console.print("Supported models:")
            for m in SUPPORTED_MODELS:
                marker = " <- current" if m == self.session.model else ""
                console.print(f"  {m}{marker}")

        elif cmd == "/list":
            self.list_sessions()

        elif cmd == "/new":
            if len(parts) != 2:
                console.print("Usage: /new <session_name>")
            else:
                self.session.save()
                self.session = Session(name=parts[1], model=self.session.model)
                console.print(f"[new session '{self.session.name}' started]")

        elif cmd == "/switch":
            if len(parts) == 1:
                from pathlib import Path

                session_files = sorted(Session.SESSIONS_DIR.glob(f"*{Session.FILENAME_SUFFIX}"))
                session_names = [f.stem for f in session_files if f.stem != self.session.name]
                selection = self._interactive_picker(
                    "Switch to session:", session_names, current=self.session.name
                )
                if selection:
                    try:
                        self.session.save()
                        self.session = Session.load(selection)
                        console.print(
                            f"[switched to session '{self.session.name}'] (model={self.session.model})"
                        )
                    except FileNotFoundError as exc:
                        console.print(exc)
                return True

            if len(parts) != 2:
                console.print("Usage: /switch <session_name>")
            else:
                try:
                    self.session.save()
                    self.session = Session.load(parts[1])
                    console.print(
                        f"[switched to session '{self.session.name}'] (model={self.session.model})"
                    )
                except FileNotFoundError as exc:
                    console.print(exc)

        elif cmd == "/clear":
            # Remove all turn messages but keep system prompt
            self.session.messages.clear()
            self.session.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

            # Reset any IDs cached in the wrapper so next request starts fresh
            if hasattr(self.client, "_last_response_id"):
                self.client._last_response_id = None  # pylint: disable=protected-access

            self.session.save()
            console.print("[conversation cleared – system prompt reinstated]")

        elif cmd == "/tool":
            if (
                len(parts) != 3
                or parts[1].lower() != "websearch"
                or parts[2] not in {"on", "off"}
            ):
                console.print("Usage: /tool websearch on|off")
            else:
                self.session.enable_web_search = parts[2] == "on"
                state = "enabled" if self.session.enable_web_search else "disabled"
                console.print(f"[web search tool {state}]")

        elif cmd == "/reasoning":
            if len(parts) != 2 or parts[1] not in {"on", "off"}:
                console.print("Usage: /reasoning on|off")
            else:
                self.session.enable_reasoning_summary = parts[1] == "on"
                state = "enabled" if self.session.enable_reasoning_summary else "disabled"
                console.print(f"[reasoning summaries {state}]")

        elif cmd == "/delete":
            if len(parts) == 1:
                from pathlib import Path

                session_files = sorted(Session.SESSIONS_DIR.glob(f"*{Session.FILENAME_SUFFIX}"))
                session_names = [f.stem for f in session_files if f.stem != self.session.name]
                selection = self._interactive_picker("Delete session:", session_names)
                if not selection:
                    return True
                target = selection
            elif len(parts) == 2:
                target = parts[1]
            else:
                console.print("Usage: /delete <session_name>")
                return True

            if target == self.session.name:
                console.print(
                    Ansi.style(
                        "Cannot delete the session you are currently using. Switch to another session first.",
                        Ansi.FG_RED,
                    )
                )
                return True

            path = Session.SESSIONS_DIR / f"{target}{Session.FILENAME_SUFFIX}"
            if not path.exists():
                console.print(f"Session '{target}' does not exist.")
                return True

            try:
                path.unlink()
                console.print(f"[session '{target}' deleted]")
            except OSError as exc:
                console.print(Ansi.style(f"Failed to delete session '{target}': {exc}", Ansi.FG_RED))
            return True

        else:
            console.print(Ansi.style(f"Unknown command: {cmd} (see /help)", Ansi.FG_RED))

        return True

    # ---------------- Interaction loop ---------------

    def repl(self) -> None:
        """Run the interactive read–eval–print-loop."""
        console.print(Panel.fit("OpenAI Chat CLI", style="bold magenta"))

        console.print(
            Ansi.style("Type your message and press Enter. Commands start with '/'.", Ansi.FG_YELLOW),
            Ansi.style(f"Current model: {self.session.model}.", Ansi.FG_YELLOW),
            Ansi.style("Type /help for help.", Ansi.FG_YELLOW),
            sep="\n",
        )

        while True:
            try:
                line = console.input(f"{USER_LABEL}> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[signal caught – exiting]")
                self.session.save()
                break

            if not line:
                continue

            if line.startswith("/"):
                if not self.handle_command(line):
                    break
                continue

            # Add user message & save early
            self.session.add_user_message(line)
            self.session.save()

            assistant_content = self.client.chat_completion(
                model=self.session.model,
                messages=self.session.messages,
                enable_web_search=self.session.enable_web_search,
                enable_reasoning_summary=self.session.enable_reasoning_summary,
            )
            self.session.add_assistant_message(assistant_content)
            self.session.save()


# ---------------------------------------------------------------------------
# Entrypoint helpers (keeping it separate simplifies __main__ handling)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Interactive CLI for OpenAI chat models with session support."
    )
    parser.add_argument("--session", "-s", help="Session name (default: 'default')", default="default")
    parser.add_argument("--model", "-m", help="Model name to use (overrides saved value)")
    return parser.parse_args()


def run_cli() -> None:  # pragma: no cover
    args = _parse_args()

    # ------------------------------------------------------------------
    # Load or create session
    # ------------------------------------------------------------------
    try:
        session = Session.load(args.session)
        if args.model:
            session.model = args.model
    except FileNotFoundError:
        default_model = args.model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
        if default_model not in SUPPORTED_MODELS:
            console.print(
                f"Warning: model '{default_model}' is not in the supported list. "
                "Falling back to default 'gpt-4o'."
            )
            default_model = "gpt-4o"
        session = Session(name=args.session, model=default_model)

    # ------------------------------------------------------------------
    # Configure OpenAI SDK
    # ------------------------------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: attempt to read from ~/.zshrc (convenience for macOS users)
        zshrc_path = Path.home() / ".zshrc"
        if zshrc_path.exists():
            rc_text = zshrc_path.read_text()
            import re

            pattern = re.compile(r"(?:export\s+)?OPENAI_API_KEY\s*=\s*['\"]?([^'\"\n]+)['\"]?")
            match = pattern.search(rc_text)
            if match:
                api_key = match.group(1).strip()
                os.environ["OPENAI_API_KEY"] = api_key  # inject for downstream

        if not api_key:
            sys.stderr.write(
                "Error: OPENAI_API_KEY environment variable is not set.\n"
                "(Tried reading from environment and ~/.zshrc)\n"
            )
            sys.exit(1)

    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
    wrapper = OpenAIClientWrapper(client)

    ChatCLI(session, wrapper).repl()


if __name__ == "__main__":  # pragma: no cover
    run_cli() 