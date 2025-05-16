"""Simple interactive CLI for chatting with OpenAI models.

Features
--------
1. Session persistence: every conversation is stored on disk and can be resumed later.
2. Model switching: change the model mid-conversation with the `/model` command (or via `--model`).
3. Built-in web search tool: enable or disable it with `/tool websearch on|off`. When enabled, the request is
   sent with the official `web_search_preview` tool so the assistant can decide when to call it.

Run `python -m chat_cli` or use `chat_cli/cli.py` as the entry point.
"""
# Re-export useful symbols for convenience
from .core import Session, SUPPORTED_MODELS, SYSTEM_PROMPT
from .core.client import OpenAIClientWrapper
from .cli import ChatCLI, run_cli

__all__ = [
    "Session",
    "SUPPORTED_MODELS",
    "SYSTEM_PROMPT",
    "OpenAIClientWrapper",
    "ChatCLI",
    "run_cli",
] 