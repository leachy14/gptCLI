# Chat CLI

Lightweight command‑line interface for chatting with OpenAI models.

```
python chat_cli.py [-s SESSION] [-m MODEL]
```

Key features

* **Session persistence** – conversations are automatically saved to `~/.chat_cli_sessions/<name>.json` and can be resumed later with `--session` or the `/switch` command.
* **Model switching** – restricted to the official list below. Change models on the fly with `/model <name>` or start with `--model`.

Supported models

```
gpt-4.1
gpt-4.1-mini
gpt-4o   (default)
o1
o3
o4-mini
o3-mini
```
* **Built‑in tools** – enable the official web‑search tool with `/tool websearch on` so the model can decide when to call it.
* **Slash commands** – `/help`, `/exit`, `/list`, `/new`, `/clear`, …

Environment variables

* `OPENAI_API_KEY` – your API key (required).
* `OPENAI_BASE_URL` – custom base‑url if you proxy the API (optional).

The CLI streams the assistant’s answer in real time and works with any model that supports the Chat Completions API.

---

This file documents only the public interface; see `chat_cli.py` for implementation details.
