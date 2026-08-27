"""Groq LLM provider for orchestrator — fallback for when HF credits are exhausted.

Uses qwen/qwen3.8-27b via Groq API with native tool calling.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from orchestrator_service.domain.models import ConversationMessage, MessageRole
from orchestrator_service.domain.ports import LLMProvider, LLMResponse

logger = logging.getLogger("orchestrator.groq_llm")


class GroqLLMProvider(LLMProvider):
    """Qwen3.8-27B via Groq API — fallback LLM for orchestrator."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = "qwen/qwen3.8-27b",
        timeout_seconds: int = 60,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise RuntimeError("groq package required: pip install groq")
            self._client = Groq(api_key=self._api_key)
        return self._client

    def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        api_messages = self._build_api_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug(
            "Groq LLM call: %d messages, %d tools, model=%s",
            len(api_messages), len(tools) if tools else 0, self._model_id,
        )

        try:
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = msg.content or ""

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input": getattr(response.usage, "prompt_tokens", 0),
                    "output": getattr(response.usage, "completion_tokens", 0),
                }

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                reasoning="",
                model=response.model or self._model_id,
                token_usage=usage,
            )

        except Exception as e:
            logger.exception("Groq LLM inference failed")
            return LLMResponse(
                content=f"Error: {e}",
                tool_calls=[],
                model=self._model_id,
            )

    def health(self) -> dict[str, Any]:
        try:
            client = self._get_client()
            client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": "Say ok"}],
                max_tokens=5,
            )
            return {"status": "healthy", "model": self._model_id, "provider": "groq"}
        except Exception as e:
            return {"status": "unhealthy", "model": self._model_id, "error": str(e)}

    def _build_api_messages(self, messages: list[ConversationMessage]) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                api_messages.append({"role": "system", "content": msg.content})
            elif msg.role == MessageRole.USER:
                api_messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                api_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": (
                                    _json.dumps(tc.arguments)
                                    if isinstance(tc.arguments, dict)
                                    else tc.arguments
                                ),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                api_messages.append(api_msg)
            elif msg.role == MessageRole.TOOL:
                content = msg.content
                if not content and msg.metadata.get("result"):
                    content = _json.dumps(msg.metadata["result"], default=str)
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": content[:8000],
                })
        return api_messages
