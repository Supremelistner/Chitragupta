"""Qwen3.8-27B LLM provider adapter.

Uses HuggingFace Inference API with native tool calling.
The model has thinking mode enabled — reasoning goes to reasoning_content,
clean output (tool calls or text) goes to content.
"""
from __future__ import annotations

import logging
from typing import Any

from orchestrator_service.domain.models import ConversationMessage, MessageRole
from orchestrator_service.domain.ports import LLMProvider, LLMResponse

logger = logging.getLogger("orchestrator.llm")


class QwenLLMProvider(LLMProvider):
    """Qwen3.8-27B via HuggingFace Inference API."""

    def __init__(
        self,
        *,
        token: str,
        model_id: str = "Qwen/Qwen3.8-27B",
        timeout_seconds: int = 60,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self._token = token
        self._model_id = model_id
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self._token)
        return self._client

    def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Build OpenAI-compatible messages
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
            kwargs["temperature"] = temperature if temperature is not None else self._temperature
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["top_p"] = 0.95

        logger.debug(
            "LLM call: %d messages, %d tools, model=%s",
            len(api_messages),
            len(tools) if tools else 0,
            self._model_id,
        )

        try:
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            # Extract reasoning from thinking mode
            reasoning = ""
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                reasoning = msg.reasoning_content

            content = msg.content or ""

            # Extract tool calls
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

            # Token usage
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input": getattr(response.usage, "prompt_tokens", 0),
                    "output": getattr(response.usage, "completion_tokens", 0),
                }

            logger.debug(
                "LLM response: content=%d chars, tool_calls=%d, reasoning=%d chars",
                len(content),
                len(tool_calls),
                len(reasoning),
            )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                reasoning=reasoning,
                model=response.model or self._model_id,
                token_usage=usage,
            )

        except Exception as e:
            logger.exception("LLM inference failed")
            return LLMResponse(
                content=f"Error: {e}",
                tool_calls=[],
                model=self._model_id,
            )

    def health(self) -> dict[str, Any]:
        try:
            client = self._get_client()
            # Quick test call
            response = client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": "Say ok"}],
                max_tokens=5,
            )
            return {
                "status": "healthy",
                "model": self._model_id,
                "provider": "huggingface",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self._model_id,
                "error": str(e),
            }

    def _build_api_messages(
        self, messages: list[ConversationMessage]
    ) -> list[dict[str, Any]]:
        """Convert domain messages to OpenAI API format."""
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                api_messages.append({"role": "system", "content": msg.content})

            elif msg.role == MessageRole.USER:
                api_messages.append({"role": "user", "content": msg.content})

            elif msg.role == MessageRole.ASSISTANT:
                api_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
                # Include tool calls if present
                if msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": (
                                    __import__("json").dumps(tc.arguments)
                                    if isinstance(tc.arguments, dict)
                                    else tc.arguments
                                ),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                api_messages.append(api_msg)

            elif msg.role == MessageRole.TOOL:
                # Tool result message
                import json as _json
                content = msg.content
                if not content and msg.metadata.get("result"):
                    content = _json.dumps(msg.metadata["result"], default=str)

                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": content[:8000],  # Truncate large results
                })

        return api_messages
