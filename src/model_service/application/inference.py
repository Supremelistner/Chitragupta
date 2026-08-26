"""Inference application service — orchestrates provider calls.

This service sits between the adapters (MCP/HTTP) and the provider adapters.
It handles provider selection, request validation, and result formatting.
"""

from __future__ import annotations

import logging
from typing import Any

from model_service.domain.models import (
    InferenceRequest,
    InferenceResult,
    InferenceTaskType,
    ModelProviderType,
    ProviderHealth,
)
from model_service.domain.ports import ModelProvider, ModelRepository

logger = logging.getLogger("model_service.inference")


class InferenceService:
    """Application service for running inference through swappable providers."""

    def __init__(
        self,
        *,
        providers: dict[str, ModelProvider],
        repository: ModelRepository,
    ) -> None:
        self._providers = providers
        self._repository = repository

    def _active_provider(self) -> ModelProvider:
        name = self._repository.get_active_provider()
        provider = self._providers.get(name)
        if provider is None:
            available = list(self._providers.keys())
            raise ValueError(
                f"Provider '{name}' not available. Configured: {available}"
            )
        return provider

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference using the currently active provider."""
        provider = self._active_provider()
        logger.info(
            "Inference request: task=%s provider=%s model=%s",
            request.task.value,
            provider.provider_type.value,
            request.model_id or "default",
        )
        return provider.infer(request)

    def infer_with_provider(
        self, request: InferenceRequest, provider_name: str
    ) -> InferenceResult:
        """Run inference with a specific provider (override active selection)."""
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Provider '{provider_name}' not available")
        return provider.infer(request)

    def list_providers(self) -> list[dict[str, Any]]:
        """List all configured providers and their status."""
        results = []
        for name, provider in self._providers.items():
            health = provider.health()
            results.append({
                "name": name,
                "type": provider.provider_type.value,
                "healthy": health.healthy,
                "models": [
                    {"id": m.model_id, "display_name": m.display_name}
                    for m in health.available_models
                ],
                "error": health.error_message,
            })
        return results

    def set_active_provider(self, provider_name: str) -> dict[str, str]:
        """Switch the active provider."""
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        self._repository.set_active_provider(provider_name)
        return {
            "active_provider": provider_name,
            "status": "switched",
        }

    def get_active_provider(self) -> str:
        return self._repository.get_active_provider()

    def health(self) -> dict[str, Any]:
        """Health check across all providers."""
        active = self._repository.get_active_provider()
        provider = self._providers.get(active)
        if provider is None:
            return {"status": "degraded", "active_provider": active, "error": "Active provider not found"}
        h = provider.health()
        return {
            "status": "healthy" if h.healthy else "degraded",
            "active_provider": active,
            "healthy": h.healthy,
            "error": h.error_message,
        }
