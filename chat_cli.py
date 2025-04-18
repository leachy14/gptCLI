#!/usr/bin/env python3

"""Simple interactive CLI for chatting with OpenAI models.

Features
--------
1. Session persistence: every conversation is stored on disk and can be resumed later.
2. Model switching: change the model mid‑conversation with the `/model` command (or via `--model`).
3. Built‑in web search tool: enable or disable it with `/tool websearch on|off`. When enabled, the request is
   sent with the official `web_search_preview` tool so the assistant can decide when to call it.

Usage
-----
    python chat_cli.py [--session SESSION_NAME] [--model MODEL]

Slash commands (enter them as a line at the prompt):

    /help                       – show this help
    /exit                       – terminate the program (session is saved automatically)
    /model MODEL_NAME           – switch to a different model (e.g. /model gpt-4o-mini)
    /list                       – list available saved sessions
    /new  SESSION_NAME          – start a brand‑new session
    /switch SESSION_NAME        – switch to an existing saved session
    /clear                      – delete all messages in the current session
    /tool websearch on|off      – enable or disable the web search tool for the session
    /reasoning on|off          – show or hide reasoning summaries (reasoning models only)

Environment variables
---------------------
* OPENAI_API_KEY – your OpenAI API key (required)
* OPENAI_BASE_URL – custom base URL (optional, for self‑hosting/proxy)

The code purposefully keeps dependencies to a minimum: only `openai` and the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import readline  # noqa: F401 (improves UX by enabling history & editing)
import sys
import itertools
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

# The CLI used to print everything in the default terminal colour which made
# it difficult to visually distinguish the user input, the assistant output
# and the various status / error messages.  A few ANSI escape sequences go a
# long way towards a much nicer UX while still keeping dependencies at zero.


class _Ansi:  # pylint: disable=too-few-public-methods
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
        return "".join(codes) + text + _Ansi.RESET


# The two most common labels are used throughout the code so we pre‑compute
# them once.

USER_LABEL = _Ansi.style("you", _Ansi.FG_CYAN, _Ansi.BOLD)
ASSISTANT_LABEL = _Ansi.style("assistant", _Ansi.FG_GREEN, _Ansi.BOLD)
ERROR_LABEL = _Ansi.style("error", _Ansi.FG_RED, _Ansi.BOLD)
WARNING_LABEL = _Ansi.style("warning", _Ansi.FG_YELLOW, _Ansi.BOLD)
# Label used when printing reasoning summaries
REASONING_LABEL = _Ansi.style("reasoning", _Ansi.FG_MAGENTA, _Ansi.BOLD)


try:
    import openai
    from openai import OpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "Error: The `openai` package is not installed. Install it with `pip install --upgrade openai`\n"
    )
    raise exc


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

HOME_DIR = Path.home()
SESSIONS_DIR = HOME_DIR / ".chat_cli_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------- Supported models -----------------------------

SUPPORTED_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",  # default
    "o1",
    "o3",
    "o4-mini",
]

# ---------------------------------------------------------------------------
# System prompt (prepended to every new session)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Spinner helper (simple typing / waiting animation)
# ---------------------------------------------------------------------------


class _Spinner:
    """A minimal terminal spinner shown while waiting for the first token.

    Usage:

        spinner = _Spinner(prefix="assistant> ")
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


def _human_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


class Session:
    """Represents a chat session stored as a JSON file on disk."""

    FILENAME_SUFFIX = ".json"

    def __init__(
        self,
        name: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        enable_web_search: bool = False,
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

    # ----------------------- Disk IO ------------------------------------

    @property
    def path(self) -> Path:
        return SESSIONS_DIR / f"{self.name}{self.FILENAME_SUFFIX}"

    def save(self) -> None:
        data = {
            "model": self.model,
            "messages": self.messages,
            "enable_web_search": self.enable_web_search,
            "enable_reasoning_summary": self.enable_reasoning_summary,
            "updated_at": _human_time(),
        }
        tmp_path = self.path.with_suffix(".tmp")
        # If any unexpected non‑serializable objects slip through we coerce
        # them to strings to avoid breaking the CLI’s persistence mechanism.
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )
        tmp_path.rename(self.path)

    @classmethod
    def load(cls, name: str) -> "Session":
        path = SESSIONS_DIR / f"{name}{cls.FILENAME_SUFFIX}"
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

    # ----------------------- Conversation helpers -----------------------

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    # For function & tool call raw objects we might want to store them as‑is
    def add_raw(self, item: Dict[str, Any]) -> None:
        self.messages.append(item)


# ---------------------------------------------------------------------------
# OpenAI client wrapper
# ---------------------------------------------------------------------------


class OpenAIClientWrapper:
    """Thin wrapper that hides streaming implementation details."""

    def __init__(self, client: OpenAI):
        self.client = client
        # When using the stateful *Responses* API we need to keep track of the
        # ID of the previous response so the subsequent call can continue the
        # same thread.
        self._last_response_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_response(resp: "openai.types.responses.response.Response") -> str:  # type: ignore[name-defined]
        """Return a human‑readable text string from a Responses API object.

        The Responses API returns a rich, structured object. The assistant’s
        answer lives inside `resp.output` → `ResponseOutputMessage` →
        `ResponseOutputText`. We traverse this hierarchy and concatenate all
        text chunks (there can technically be more than one) into a single
        string so that the rest of the CLI can treat it just like a normal
        chat completion response.
        """

        try:
            # The exact classes live deep inside the OpenAI package. We avoid
            # importing them directly and instead rely on their structural
            # interface (`type` attribute) so that the code keeps working even
            # if the OpenAI SDK version changes slightly.
            texts: List[str] = []
            for output in getattr(resp, "output", []):
                if getattr(output, "type", None) != "message":
                    continue
                for content in getattr(output, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        txt = getattr(content, "text", None)
                        if isinstance(txt, str):
                            texts.append(txt)
            # Join with newlines (multiple chunks are rare but possible).
            if texts:
                return "\n".join(texts)
        except Exception:  # pragma: no cover – defensive catch‑all
            pass

        # Fallback: return the repr so the user sees *something*.
        return str(resp)

    @staticmethod
    def _extract_summary_from_response(resp: "openai.types.responses.response.Response") -> str:  # type: ignore[name-defined]
        """Return the reasoning summary text from a Responses API object if present."""

        try:
            summaries: List[str] = []
            for output in getattr(resp, "output", []):
                if getattr(output, "type", None) != "reasoning":
                    continue
                # Each reasoning item can contain multiple summary objects
                for summary_obj in getattr(output, "summary", []):
                    txt = getattr(summary_obj, "text", None)
                    if isinstance(txt, str):
                        summaries.append(txt)
            if summaries:
                return "\n".join(summaries)
        except Exception:  # pragma: no cover
            pass
        return ""

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        enable_web_search: bool,
        enable_reasoning_summary: bool,
    ) -> str:
        """Create a chat (or responses) completion and stream / print the answer.

        There are three execution paths depending on the requested features:

        1. Plain chat completions (fast & cheap) – default path.
        2. Web search enabled               → use Responses API with `tools`.
        3. Reasoning summary requested      → use Responses API with `reasoning`.

        Paths 2 and 3 are not mutually exclusive. If *either* web search or
        reasoning summary is requested we route the request through the
        Responses API and set the relevant parameters.
        """

        # ------------------------------------------------------------------
        # Path 1 – Neither web search nor reasoning summary requested
        #           → use Chat Completions (with streaming)
        # ------------------------------------------------------------------
        if not enable_web_search and not enable_reasoning_summary:
            params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": True,
            }

            accumulator: List[str] = []

            prefix = f"{ASSISTANT_LABEL}> "
            spinner = _Spinner(prefix=prefix)
            first_token_received = False

            try:
                spinner.start()
                response = self.client.chat.completions.create(**params)  # type: ignore[arg-type]
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if not delta.content:
                        continue

                    if not first_token_received:
                        # Stop spinner the moment the first actual token arrives
                        spinner.stop()
                        first_token_received = True

                    # Print token right after prefix (no extra spaces)
                    print(delta.content, end="", flush=True)
                    accumulator.append(delta.content)

                if not first_token_received:
                    # Edge case: no content returned – stop spinner anyway
                    spinner.stop()
                print()  # newline after streaming finished
            except openai.OpenAIError as e:
                spinner.stop()
                print(f"\n[{ERROR_LABEL}] OpenAI API error: {e}\n")
                return ""
            except KeyboardInterrupt:
                spinner.stop()
                print("\n[interrupted]")
                return ""

            return "".join(accumulator)

        # ------------------------------------------------------------------
        # Path 2 – Use Responses API (web search and/or reasoning summary)
        # ------------------------------------------------------------------

        try:
            # For the Responses API we only send the latest user message as the
            # `input`. The API itself is stateful and will remember the prior
            # turns, so we don't need to (and shouldn't) resend the full
            # conversation every time.

            last_user_msg = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            if last_user_msg is None:
                raise ValueError("No user message found in conversation.")

            resp_kwargs: Dict[str, Any] = {
                "model": model,
                "input": last_user_msg,
            }

            if enable_web_search:
                # Use the preview web search tool – stable models may use "web_search"
                resp_kwargs["tools"] = [{"type": "web_search_preview"}]

            if enable_reasoning_summary:
                resp_kwargs["reasoning"] = {"effort": "medium", "summary": "auto"}

            if self._last_response_id is not None:
                resp_kwargs["previous_response_id"] = self._last_response_id

            prefix = f"{ASSISTANT_LABEL}> "
            spinner = _Spinner(prefix=prefix)
            spinner.start()

            try:
                resp = self.client.responses.create(**resp_kwargs)  # type: ignore[arg-type]
            finally:
                # Ensure spinner always stops, even on exceptions.
                spinner.stop()

            # Cache for the next turn
            self._last_response_id = resp.id

            # Extract answer & (optionally) reasoning summary
            aggregated_text = self._extract_text_from_response(resp)
            aggregated_summary = (
                self._extract_summary_from_response(resp) if enable_reasoning_summary else ""
            )

            # Print the assistant's answer (prefix already on the line).
            print(f"{aggregated_text}")

            # Then print summary, if any
            if aggregated_summary:
                print(f"{REASONING_LABEL}> {aggregated_summary}")

            return aggregated_text

        except openai.BadRequestError as e:
            # The selected model might not support one of the requested features.
            unsupported_features: List[str] = []
            if enable_web_search:
                unsupported_features.append("web search")
            if enable_reasoning_summary:
                unsupported_features.append("reasoning summary")

            feature_list = " and ".join(unsupported_features) or "requested feature"
            print(
                f"\n[{WARNING_LABEL}] {feature_list.capitalize()} failed or is not supported for model '{model}': {e}. "
                "Retrying without unsupported features…\n"
            )

            return self.chat_completion(
                model=model,
                messages=messages,
                enable_web_search=False,
                enable_reasoning_summary=False,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            return ""


# ---------------------------------------------------------------------------
# CLI logic
# ---------------------------------------------------------------------------


class ChatCLI:
    def __init__(self, initial_session: Session, client_wrapper: OpenAIClientWrapper):
        self.session = initial_session
        self.client = client_wrapper

    # ---------------- Utility ----------------

    @staticmethod
    def list_sessions() -> None:
        files = sorted(SESSIONS_DIR.glob(f"*{Session.FILENAME_SUFFIX}"))
        if not files:
            print("(no saved sessions)")
            return
        print(_Ansi.style("Saved sessions:", _Ansi.BOLD, _Ansi.FG_MAGENTA))
        for file in files:
            name = file.stem
            updated = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            indicator_char = "★" if file == (SESSIONS_DIR / f"{name}{Session.FILENAME_SUFFIX}") else " "
            colour = _Ansi.FG_GREEN if indicator_char == "★" else _Ansi.FG_CYAN
            label = _Ansi.style(name, colour)
            print(f"  {indicator_char} {label} (updated: {updated})")

    # ---------------- Command handling ---------------

    def handle_command(self, line: str) -> bool:
        """Returns True if the CLI should continue the REPL, False to exit."""

        parts = line.strip().split()
        if not parts:
            return True

        cmd = parts[0].lower()

        if cmd == "/help":
            print(__doc__)
        elif cmd == "/exit":
            self.session.save()
            print("Session saved. Bye!")
            return False
        elif cmd == "/model":
            if len(parts) != 2:
                print("Usage: /model <model_name>")
            else:
                model_name = parts[1]
                if model_name not in SUPPORTED_MODELS:
                    print("Unsupported model. Use /models to see the list of supported models.")
                else:
                    self.session.model = model_name
                    print(f"[model switched to {self.session.model}]")
        elif cmd == "/models":
            print("Supported models:")
            for m in SUPPORTED_MODELS:
                marker = " <- current" if m == self.session.model else ""
                print(f"  {m}{marker}")
        elif cmd == "/list":
            self.list_sessions()
        elif cmd == "/new":
            if len(parts) != 2:
                print("Usage: /new <session_name>")
            else:
                self.session.save()
                self.session = Session(name=parts[1], model=self.session.model)
                print(f"[new session '{self.session.name}' started]")
        elif cmd == "/switch":
            if len(parts) != 2:
                print("Usage: /switch <session_name>")
            else:
                try:
                    self.session.save()
                    self.session = Session.load(parts[1])
                    print(f"[switched to session '{self.session.name}'] (model={self.session.model})")
                except FileNotFoundError as exc:
                    print(exc)
        elif cmd == "/clear":
            self.session.messages.clear()
            print("[conversation cleared]")
        elif cmd == "/tool":
            if len(parts) != 3 or parts[1].lower() != "websearch" or parts[2] not in {"on", "off"}:
                print("Usage: /tool websearch on|off")
            else:
                self.session.enable_web_search = parts[2] == "on"
                state = "enabled" if self.session.enable_web_search else "disabled"
                print(f"[web search tool {state}]")
        elif cmd == "/reasoning":
            if len(parts) != 2 or parts[1] not in {"on", "off"}:
                print("Usage: /reasoning on|off")
            else:
                self.session.enable_reasoning_summary = parts[1] == "on"
                state = "enabled" if self.session.enable_reasoning_summary else "disabled"
                print(f"[reasoning summaries {state}]")
        else:
            print(_Ansi.style(f"Unknown command: {cmd} (see /help)", _Ansi.FG_RED))

        return True

    # ---------------- Interaction loop ---------------

    def repl(self) -> None:
        # ------------------------------------------------------------------
        # Decorative banner
        # ------------------------------------------------------------------

        banner_lines = [
            _Ansi.style(" ┌──────────────────────────────────────────────┐", _Ansi.FG_MAGENTA),
            _Ansi.style(" │         OpenAI   Chat   CLI                │", _Ansi.FG_MAGENTA, _Ansi.BOLD),
            _Ansi.style(" └──────────────────────────────────────────────┘", _Ansi.FG_MAGENTA),
        ]

        print("\n".join(banner_lines))

        # Usage blurb ---------------------------------------------------
        print(
            _Ansi.style("Type your message and press Enter. Commands start with '/'.", _Ansi.FG_YELLOW),
            _Ansi.style(f"Current model: {self.session.model}.", _Ansi.FG_YELLOW),
            _Ansi.style("Type /help for help.", _Ansi.FG_YELLOW),
            sep="\n",
        )

        while True:
            try:
                line = input(f"{USER_LABEL}> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl‑D or Ctrl‑C -> exit politely
                print("\n[signal caught – exiting]")
                self.session.save()
                break

            if not line:
                continue

            if line.startswith("/"):
                if not self.handle_command(line):
                    break
                continue

            # Normal user message flow -----------------------------------
            self.session.add_user_message(line)
            self.session.save()  # save early so we don't lose user message on crash

            assistant_content = self.client.chat_completion(
                model=self.session.model,
                messages=self.session.messages,
                    enable_web_search=self.session.enable_web_search,
                    enable_reasoning_summary=self.session.enable_reasoning_summary,
            )

            self.session.add_assistant_message(assistant_content)
            self.session.save()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Interactive CLI for OpenAI chat models with session support.")
    parser.add_argument("--session", "-s", help="Session name (default: 'default')", default="default")
    parser.add_argument("--model", "-m", help="Model name to use (overrides saved value)")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()

    try:
        session = Session.load(args.session)
        if args.model:
            session.model = args.model
    except FileNotFoundError:
        default_model = args.model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
        # Validate default model value
        if default_model not in SUPPORTED_MODELS:
            print(
                f"Warning: model '{default_model}' is not in the supported list. "
                "Falling back to default 'gpt-4o'."
            )
            default_model = "gpt-4o"

        session = Session(name=args.session, model=default_model)

    # Setup OpenAI client --------------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: try to read it from ~/.zshrc (common for macOS users)
        import re

        zshrc_path = Path.home() / ".zshrc"
        if zshrc_path.exists():
            rc_text = zshrc_path.read_text()
            # Match both `export OPENAI_API_KEY="sk-..."` and `OPENAI_API_KEY=sk-...`
            pattern = re.compile(r"(?:export\s+)?OPENAI_API_KEY\s*=\s*['\"]?([^'\"\n]+)['\"]?")
            match = pattern.search(rc_text)
            if match:
                api_key = match.group(1).strip()
                # Inject into environment for downstream libraries
                os.environ["OPENAI_API_KEY"] = api_key

        if not api_key:
            sys.stderr.write(
                "Error: OPENAI_API_KEY environment variable is not set.\n"
                "(Tried reading from environment and ~/.zshrc)\n"
            )
            sys.exit(1)

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
    wrapper = OpenAIClientWrapper(client)

    ChatCLI(session, wrapper).repl()


if __name__ == "__main__":  # pragma: no cover
    main()
