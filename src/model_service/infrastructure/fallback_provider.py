"""Fallback provider wrapper - tries primary (HF), falls back to secondary (Groq).

Transparent to the rest of the system. Both providers implement the same interface.
"""

from __future__ import annotations

import logging
from typing import Any

from dataclasses import replace
from model_service.domain.models import (
    InferenceRequest,
    InferenceResult,
    ModelInfo,
    ModelProviderType,
    ProviderHealth,
)
from model_service.domain.ports import ModelProvider

logger = logging.getLogger("model_service.fallback")

# HTTP statuses worth retrying on the secondary provider.
_RECOVERABLE_STATUSES = frozenset({402, 408, 425, 429, 500, 502, 503, 504})
# Definitive client errors: retrying elsewhere won't help (bad key,
# unknown model, malformed request). Anything unlisted falls through
# to the legacy keyword match below.
_FATAL_STATUSES = frozenset({400, 401, 403, 404, 405, 422})

# Legacy keyword fallback for errors without a machine-readable status.
_RECOVERABLE_KEYWORDS = (
    "402", "429", "payment required", "rate limit",
    "quota", "credit", "insufficient", "timeout",
    "connection", "unreachable", "service unavailable",
    "503", "504", "gateway",
)


def _is_recoverable(result: InferenceResult) -> bool:
    """True iff a failed primary result should trigger the fallback.

    Prefers the structured ``http_status`` providers attach to error
    metadata; falls back to keyword matching for unclassified errors
    so behavior never regresses to fail-closed.
    """
    status = (result.metadata or {}).get("http_status")
    if isinstance(status, int):
        return status in _RECOVERABLE_STATUSES
    error_msg = (result.output or "").lower()
    return any(kw in error_msg for kw in _RECOVERABLE_KEYWORDS)


class FallbackProvider(ModelProvider):
    """Tries primary provider first; on failure (402, 429, timeout), uses fallback."""

    def __init__(self, primary: ModelProvider, fallback: ModelProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def provider_type(self) -> ModelProviderType:
        return self._primary.provider_type

    def ping(self) -> None:
        try:
            self._primary.ping()
        except Exception:
            logger.warning("Primary provider ping failed, trying fallback")
            self._fallback.ping()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        try:
            result = self._primary.infer(request)
        except Exception as primary_exc:
            # Primary raised rather than returning an ERROR: result. This
            # happens for connection errors, timeouts, or import errors.
            logger.warning(
                "Primary provider raised %s (%s) — falling back to secondary",
                primary_exc.__class__.__name__, primary_exc,
            )
            fallback_result = self._fallback.infer(request)
            meta = dict(fallback_result.metadata or {})
            meta["used_fallback"] = True
            meta["primary_error"] = f"{primary_exc.__class__.__name__}: {primary_exc}"
            return replace(fallback_result, metadata=meta)

        if result.output and result.output.startswith("ERROR:"):
            if _is_recoverable(result):
                logger.warning(
                    "Primary provider failed (%s), falling back to secondary",
                    result.output[:100],
                )
                fallback_result = self._fallback.infer(request)
                meta = dict(fallback_result.metadata or {})
                meta["used_fallback"] = True
                meta["primary_error"] = result.output[:200]
                return replace(fallback_result, metadata=meta)

        return result

    def list_models(self) -> list[ModelInfo]:
        # Copy: extending the primary's own list object would mutate the
        # provider's internal state for every caller holding that reference.
        return [*self._primary.list_models(), *self._fallback.list_models()]

    def health(self) -> ProviderHealth:
        primary_health = self._primary.health()
        fallback_health = self._fallback.health()
        return ProviderHealth(
            provider=self.provider_type,
            healthy=primary_health.healthy or fallback_health.healthy,
            available_models=(
                primary_health.available_models or ()
            ) + (
                fallback_health.available_models or ()
            ),
            error_message=(
                None if primary_health.healthy
                else f"Primary: {primary_health.error_message}; Fallback healthy"
            ),
        )

    def close(self) -> None:
        self._primary.close()
        self._fallback.close()
