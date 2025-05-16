"""OpenAI client wrapper for chat completions and Responses API handling."""

from __future__ import annotations

import sys
from typing import List, Dict, Any, Optional

import openai
from openai import OpenAI  # type: ignore

from ..utils import (
    ASSISTANT_LABEL,
    ERROR_LABEL,
    WARNING_LABEL,
    REASONING_LABEL,
    Spinner,
)


class OpenAIClientWrapper:
    """Thin wrapper around the OpenAI Python SDK hiding streaming details."""

    def __init__(self, client: OpenAI):
        self.client = client
        # When using the stateful *Responses* API we need to keep track of the
        # ID of the previous response so subsequent calls can continue the
        # same thread.
        self._last_response_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_response(
        resp: "openai.types.responses.response.Response",  # type: ignore[name-defined]
    ) -> str:
        """Return a human-readable text string from a Responses API object."""
        try:
            texts: List[str] = []
            for output in getattr(resp, "output", []):
                if getattr(output, "type", None) != "message":
                    continue
                for content in getattr(output, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        txt = getattr(content, "text", None)
                        if isinstance(txt, str):
                            texts.append(txt)
            if texts:
                return "\n".join(texts)
        except Exception:  # pragma: no cover – defensive catch-all
            pass
        # Fallback: return the repr so the user sees *something*.
        return str(resp)

    @staticmethod
    def _extract_summary_from_response(
        resp: "openai.types.responses.response.Response",  # type: ignore[name-defined]
    ) -> str:
        """Return the reasoning summary text from a Responses API object if present."""
        try:
            summaries: List[str] = []
            for output in getattr(resp, "output", []):
                if getattr(output, "type", None) != "reasoning":
                    continue
                for summary_obj in getattr(output, "summary", []):
                    txt = getattr(summary_obj, "text", None)
                    if isinstance(txt, str):
                        summaries.append(txt)
            if summaries:
                return "\n".join(summaries)
        except Exception:  # pragma: no cover – defensive catch-all
            pass
        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool,
        enable_reasoning_summary: bool,
    ) -> str:
        """Create a completion (chat or responses) and print the answer.

        There are three execution paths depending on the requested features:
        1. Neither web search nor reasoning summary → Chat Completions API (streaming).
        2. Web search and/or reasoning summary     → Responses API.
        """

        # ------------------------------------------------------------------
        # Path 1 – Plain chat completion with streaming
        # ------------------------------------------------------------------
        if not enable_web_search and not enable_reasoning_summary:
            params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": True,
            }

            accumulator: List[str] = []
            prefix = f"{ASSISTANT_LABEL}> "
            spinner = Spinner(prefix=prefix)
            first_token_received = False

            try:
                spinner.start()
                response = self.client.chat.completions.create(**params)  # type: ignore[arg-type]
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if not delta.content:
                        continue

                    if not first_token_received:
                        spinner.stop()
                        first_token_received = True

                    print(delta.content, end="", flush=True)
                    accumulator.append(delta.content)

                if not first_token_received:
                    spinner.stop()
                print()  # new line after stream ends
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
        # Path 2 – Responses API (web search and/or reasoning summary)
        # ------------------------------------------------------------------
        try:
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

            system_prompts = [m.get("content", "") for m in messages if m.get("role") == "system"]
            if system_prompts:
                resp_kwargs["instructions"] = "\n\n".join(system_prompts)

            if enable_web_search:
                resp_kwargs["tools"] = [{"type": "web_search_preview"}]

            if enable_reasoning_summary:
                resp_kwargs["reasoning"] = {"effort": "medium", "summary": "auto"}

            if self._last_response_id is not None:
                resp_kwargs["previous_response_id"] = self._last_response_id

            prefix = f"{ASSISTANT_LABEL}> "
            spinner = Spinner(prefix=prefix)
            spinner.start()
            try:
                resp = self.client.responses.create(**resp_kwargs)  # type: ignore[arg-type]
            finally:
                spinner.stop()

            self._last_response_id = resp.id  # cache for next turn

            aggregated_text = self._extract_text_from_response(resp)
            aggregated_summary = (
                self._extract_summary_from_response(resp) if enable_reasoning_summary else ""
            )

            print(f"{aggregated_text}")
            if aggregated_summary:
                print(f"{REASONING_LABEL}> {aggregated_summary}")

            return aggregated_text

        except openai.BadRequestError as e:
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