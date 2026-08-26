"""Port interfaces for the Model Service.

These Protocol classes define the contracts that provider adapters must implement.
Swapping providers means implementing a new adapter — no application or domain
code changes required.
"""

from __future__ import annotations

from typing import Protocol, Any

from model_service.domain.models import (
    InferenceRequest,
    InferenceResult,
    ModelInfo,
    ModelProviderType,
    ProviderHealth,
)


class ModelProvider(Protocol):
    """Core protocol for a model inference backend.

    Every provider adapter (HuggingFace, Fireworks, OpenAI, etc.)
    must implement this interface.
    """

    @property
    def provider_type(self) -> ModelProviderType:
        """Return the provider type identifier."""
        ...

    def ping(self) -> None:
        """Verify the provider is reachable."""
        ...

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on the given request."""
        ...

    def list_models(self) -> list[ModelInfo]:
        """List available models from this provider."""
        ...

    def health(self) -> ProviderHealth:
        """Return provider health status."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...


class PromptTemplate(Protocol):
    """Protocol for prompt templates (to be designed later).

    Templates are keyed by InferenceTaskType and can be versioned.
    """

    def render(self, task: str, **kwargs: Any) -> str:
        """Render a prompt template for the given task."""
        ...

    def list_templates(self) -> list[str]:
        """List available template identifiers."""
        ...


class ModelRepository(Protocol):
    """Protocol for persisting model configuration and selection state."""

    def get_active_provider(self) -> str:
        """Return the currently active provider identifier."""
        ...

    def set_active_provider(self, provider: str) -> None:
        """Set the active provider."""
        ...

    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Return configuration for a specific provider."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
