"""Domain models for the Model Service.

These define the contract between the application layer and any model provider.
Provider adapters translate between these models and provider-specific APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ModelProviderType(str, Enum):
    """Supported model provider backends."""
    HUGGINGFACE = "huggingface"
    FIREWORKS = "fireworks"
    DEEPINFRA = "deepinfra"
    OPENAI = "openai"
    GEMINI = "gemini"
    LOCAL = "local"


class InferenceTaskType(str, Enum):
    """Types of inference tasks the model service can perform."""
    TEXT_EXTRACTION = "text_extraction"
    DOCUMENT_CLASSIFICATION = "document_classification"
    OCR_VERIFICATION = "ocr_verification"
    METADATA_EXTRACTION = "metadata_extraction"
    CONTENT_SUMMARIZATION = "content_summarization"
    PRIVACY_CLASSIFICATION = "privacy_classification"
    CUSTOM = "custom"
    TEMPLATE_FROM_WEB = "template_from_web"
    AUDIO_SYNTHESIS = "audio_synthesis"


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """A request for model inference.

    Attributes:
        task: The type of inference task.
        image_bytes: Raw image/document bytes (optional — for vision tasks).
        image_mime_type: MIME type of the image (e.g., "image/jpeg").
        text: Text input for the task (e.g., OCR output to verify).
        prompt: The prompt template to use (empty string = to be designed).
        model_id: Optional override for which model to use.
        parameters: Task-specific parameters (temperature, max_tokens, etc.).
        request_id: Optional caller-supplied request identifier.
    """
    task: InferenceTaskType
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    text: str | None = None
    prompt: str = ""
    model_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """The result of a model inference call.

    Attributes:
        task: The task that was performed.
        provider: Which provider handled the request.
        model_id: Which model was actually used.
        output: The raw model output (text, classification, etc.).
        confidence: Optional confidence score (0.0–1.0).
        latency_ms: Time taken for the inference call.
        token_usage: Token counts if applicable (input, output).
        metadata: Provider-specific metadata.
        request_id: Echoed from the request.
        created_at: Timestamp of the result.
    """
    task: InferenceTaskType
    provider: ModelProviderType
    model_id: str
    output: str
    confidence: float | None = None
    latency_ms: float | None = None
    token_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata about an available model."""
    model_id: str
    provider: ModelProviderType
    display_name: str
    capabilities: tuple[str, ...] = ()
    max_image_size_bytes: int | None = None
    supports_vision: bool = False
    supports_text: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Health status of a model provider."""
    provider: ModelProviderType
    healthy: bool
    available_models: tuple[ModelInfo, ...] = ()
    error_message: str | None = None
    latency_ms: float | None = None
