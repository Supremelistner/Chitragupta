"""In-memory model repository for development and testing."""

from __future__ import annotations

from model_service.domain.ports import ModelRepository


class InMemoryModelRepository(ModelRepository):
    def __init__(self, initial_provider: str = "huggingface") -> None:
        self._active_provider = initial_provider
        self._configs: dict[str, dict] = {}

    def get_active_provider(self) -> str:
        return self._active_provider

    def set_active_provider(self, provider: str) -> None:
        self._active_provider = provider

    def get_provider_config(self, provider: str) -> dict:
        return self._configs.get(provider, {})

    def close(self) -> None:
        pass
