"""Groq VL provider adapter -- fallback for when HF credits are exhausted.

Uses the Groq Python SDK (OpenAI-compatible) with vision models.
Primary use: document extraction, classification, OCR verification.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from model_service.domain.models import (
    InferenceRequest,
    InferenceResult,
    InferenceTaskType,
    ModelInfo,
    ModelProviderType,
    ProviderHealth,
)
from model_service.domain.ports import ModelProvider

logger = logging.getLogger("model_service.groq")


class GroqProviderAdapter(ModelProvider):
    """Adapter for Groq inference API with vision models.

    Supports: qwen/qwen3.6-27b, qwen/qwen3.8-27b
    Free tier: 30 RPM, ~6K TPM, 14,400 req/day.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = "qwen/qwen3.6-27b",
        timeout_seconds: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    @property
    def provider_type(self) -> ModelProviderType:
        return ModelProviderType.HUGGINGFACE  # Use same type as fallback

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise RuntimeError(
                    "groq package is required. Install with: pip install groq"
                )
            self._client = Groq(api_key=self._api_key)
        return self._client

    def ping(self) -> None:
        """Verify Groq API is reachable."""
        self._get_client()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        start = time.monotonic()
        try:
            output, token_usage = self._call_inference(request)
            latency_ms = (time.monotonic() - start) * 1000
            return InferenceResult(
                task=request.task,
                provider=self.provider_type,
                model_id=self._model_id,
                output=output,
                latency_ms=latency_ms,
                token_usage=token_usage,
                request_id=request.request_id,
                metadata={"provider": "groq", "fallback": True},
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.exception("Groq inference failed")
            return InferenceResult(
                task=request.task,
                provider=self.provider_type,
                model_id=self._model_id,
                output=f"ERROR: {exc}",
                latency_ms=latency_ms,
                request_id=request.request_id,
                metadata={"error": str(exc), "provider": "groq"},
            )

    def _call_inference(self, request: InferenceRequest) -> tuple[str, dict[str, int] | None]:
        """Call the Groq API via the groq SDK."""
        client = self._get_client()
        messages = self._build_messages(request)

        response = client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=request.parameters.get("temperature", self._temperature),
            max_tokens=request.parameters.get("max_tokens", self._max_tokens),
        )

        output = self._strip_markdown_fences(response.choices[0].message.content or "")
        token_usage = None
        if hasattr(response, "usage") and response.usage:
            token_usage = {
                "input": getattr(response.usage, "prompt_tokens", 0),
                "output": getattr(response.usage, "completion_tokens", 0),
            }
        return output, token_usage

    def _build_messages(self, request: InferenceRequest) -> list[dict[str, Any]]:
        """Build OpenAI-compatible messages, including image if present."""
        messages: list[dict[str, Any]] = []

        system_prompt = self._system_prompt(request.task)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, Any]] = []

        if request.image_bytes and request.image_mime_type:
            b64_image = base64.b64encode(request.image_bytes).decode("ascii")
            data_url = f"data:{request.image_mime_type};base64,{b64_image}"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        text_parts = []
        if request.prompt:
            text_parts.append(request.prompt)
        if request.text:
            text_parts.append("Input text:\n" + request.text)
        if text_parts:
            user_content.append({
                "type": "text",
                "text": "\n\n".join(text_parts),
            })

        if not user_content:
            user_content.append({"type": "text", "text": "No input provided."})

        messages.append({"role": "user", "content": user_content})
        return messages

    def _system_prompt(self, task: InferenceTaskType) -> str:
        """Return the system prompt for a task."""
        prompts: dict[InferenceTaskType, str] = {
            InferenceTaskType.TEXT_EXTRACTION: _TEXT_EXTRACTION_PROMPT,
            InferenceTaskType.DOCUMENT_CLASSIFICATION: _CLASSIFICATION_PROMPT,
            InferenceTaskType.OCR_VERIFICATION: _OCR_VERIFICATION_PROMPT,
            InferenceTaskType.METADATA_EXTRACTION: _METADATA_EXTRACTION_PROMPT,
            InferenceTaskType.CONTENT_SUMMARIZATION: _SUMMARIZATION_PROMPT,
            InferenceTaskType.PRIVACY_CLASSIFICATION: _PRIVACY_CLASSIFICATION_PROMPT,
            InferenceTaskType.CUSTOM: _CUSTOM_PROMPT,
        }
        return prompts.get(task, "")

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=self._model_id,
                provider=self.provider_type,
                display_name=f"{self._model_id} (via Groq - fallback)",
                capabilities=("vision", "text", "multilingual", "document-understanding"),
                supports_vision=True,
                supports_text=True,
            ),
        ]

    def health(self) -> ProviderHealth:
        try:
            self.ping()
            return ProviderHealth(
                provider=self.provider_type,
                healthy=True,
                available_models=tuple(self.list_models()),
            )
        except Exception as exc:
            return ProviderHealth(
                provider=self.provider_type,
                healthy=False,
                error_message=str(exc),
            )

    def close(self) -> None:
        self._client = None

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()


# Prompts are imported from huggingface adapter to avoid duplication
from model_service.infrastructure.huggingface import (
    _TEXT_EXTRACTION_PROMPT,
    _CLASSIFICATION_PROMPT,
    _OCR_VERIFICATION_PROMPT,
    _METADATA_EXTRACTION_PROMPT,
    _SUMMARIZATION_PROMPT,
    _PRIVACY_CLASSIFICATION_PROMPT,
    _CUSTOM_PROMPT,
)
