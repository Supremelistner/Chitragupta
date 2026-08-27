"""Fallback LLM wrapper — tries primary (HF), falls back to Groq.

Transparent to the orchestrator engine.
"""

from __future__ import annotations

import logging
from typing import Any

from orchestrator_service.domain.models import ConversationMessage
from orchestrator_service.domain.ports import LLMProvider, LLMResponse

logger = logging.getLogger("orchestrator.fallback_llm")


class FallbackLLMProvider(LLMProvider):
    """Tries primary LLM first; on failure, uses Groq fallback."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        result = self._primary.chat(messages, tools, temperature, max_tokens)

        # Check if primary failed with recoverable error
        if result.content and result.content.startswith("Error:"):
            error_msg = result.content.lower()
            is_recoverable = any(
                kw in error_msg
                for kw in ["402", "429", "payment required", "rate limit",
                           "quota", "credit", "insufficient", "timeout",
                           "request limit"]
            )
            if is_recoverable:
                logger.warning(
                    "Primary LLM failed (%s), falling back to Groq",
                    result.content[:100],
                )
                return self._fallback.chat(messages, tools, temperature, max_tokens)

        return result

    def health(self) -> dict[str, Any]:
        primary = self._primary.health()
        fallback = self._fallback.health()
        return {
            "status": "healthy" if primary.get("status") == "healthy" or fallback.get("status") == "healthy" else "unhealthy",
            "primary": primary,
            "fallback": fallback,
        }
