"""Canonical data contracts shared across all microservices.

This module defines the stable IDs, status enums, and relationship models
that all four services (Document, Model, Validator, Web Search) use.

Design principles:
    1. document_id is the universal identifier — a hex UUID string.
    2. version is an integer (1, 2, 3...) — monotonically increasing per document.
    3. chunk_id is deterministic: sha256(f"{document_id}:{version}:{chunk_index}")[:16]
    4. template_id is a stable string (e.g., "aadhaar_card_v1").
    5. All IDs are strings — no integer auto-increment exposed across services.
    6. Every relationship uses these stable IDs, never database row IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Stable IDs
# ═══════════════════════════════════════════════════════════════════════════

# All IDs are plain strings.  Services must not use integer auto-increment
# IDs across service boundaries.

DocumentId = str    # Hex UUID: "a1b2c3d4e5f6..."
Version = int       # Monotonically increasing: 1, 2, 3...
ChunkId = str       # Deterministic: sha256(f"{doc_id}:{version}:{idx}")[:16]
TemplateId = str    # Stable string: "aadhaar_card_v1"
RunId = str         # Hex UUID for a validation/model-execution run
StorageKey = str    # Relative path: "documents/{doc_id}/v{version}/{filename}"


# ═══════════════════════════════════════════════════════════════════════════
# Document Lifecycle Enums
# ═══════════════════════════════════════════════════════════════════════════

class ProcessingStatus(str, Enum):
    """Document processing pipeline status — shared across Document and Model services."""
    RECEIVED = "RECEIVED"
    STORING = "STORING"
    STORED = "STORED"
    OCR_IN_PROGRESS = "OCR_IN_PROGRESS"
    OCR_COMPLETE = "OCR_COMPLETE"
    MODEL_INFERENCE_IN_PROGRESS = "MODEL_INFERENCE_IN_PROGRESS"
    MODEL_INFERENCE_COMPLETE = "MODEL_INFERENCE_COMPLETE"
    VALIDATING = "VALIDATING"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PrivacyClassification(str, Enum):
    """Document privacy levels — used by Document and Validator services."""
    OPEN = "OPEN"
    OPEN_NOT_PUBLIC = "OPEN_NOT_PUBLIC"
    PRIVATE = "PRIVATE"
    SENSITIVE = "SENSITIVE"


class FileKind(str, Enum):
    """Document file types."""
    PDF = "PDF"
    IMAGE = "IMAGE"
    UNKNOWN = "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════
# Validation Enums
# ═══════════════════════════════════════════════════════════════════════════

class ValidationStatus(str, Enum):
    """Result of document validation — used by Validator service."""
    VALID = "VALID"
    INVALID = "INVALID"
    SUSPICIOUS = "SUSPICIOUS"
    UNKNOWN_DOCUMENT = "UNKNOWN_DOCUMENT"
    ERROR = "ERROR"


class ExpiryStatus(str, Enum):
    """Temporal validity status — used by Validator service."""
    VALID = "VALID"
    EXPIRED = "EXPIRED"
    EXPIRING_SOON = "EXPIRING_SOON"
    NO_EXPIRY = "NO_EXPIRY"
    UNKNOWN = "UNKNOWN"


class AlertSeverity(str, Enum):
    """Alert severity levels — used by Validator → Document service."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ═══════════════════════════════════════════════════════════════════════════
# Model Service Enums
# ═══════════════════════════════════════════════════════════════════════════

class ModelProvider(str, Enum):
    """Supported model providers."""
    HUGGINGFACE = "huggingface"
    FIREWORKS = "fireworks"
    DEEPINFRA = "deepinfra"
    OPENAI = "openai"
    LOCAL = "local"


class InferenceTask(str, Enum):
    """Types of inference tasks the model service performs."""
    TEXT_EXTRACTION = "text_extraction"
    DOCUMENT_CLASSIFICATION = "document_classification"
    OCR_VERIFICATION = "ocr_verification"
    METADATA_EXTRACTION = "metadata_extraction"
    CONTENT_SUMMARIZATION = "content_summarization"
    PRIVACY_CLASSIFICATION = "privacy_classification"
    CUSTOM = "custom"


# ═══════════════════════════════════════════════════════════════════════════
# Relationship Models
# ═══════════════════════════════════════════════════════════════════════════

class RelationshipType(str, Enum):
    """Document-to-document relationships."""
    REQUIRED_FOR = "REQUIRED_FOR"
    SUPPORTS = "SUPPORTS"
    PROVES = "PROVES"
    REFERENCES = "REFERENCES"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    DEPENDS_ON = "DEPENDS_ON"
    COMPLETES = "COMPLETES"


# ═══════════════════════════════════════════════════════════════════════════
# Core Data Contracts
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    """Canonical document identity — used by all services."""
    document_id: DocumentId
    version: Version


@dataclass(frozen=True, slots=True)
class ChunkIdentity:
    """Canonical chunk identity — links Qdrant points to PostgreSQL records."""
    chunk_id: ChunkId
    document_id: DocumentId
    version: Version
    chunk_index: int


@dataclass(frozen=True, slots=True)
class StorageReference:
    """File storage reference — PostgreSQL stores this, never the binary."""
    storage_key: StorageKey
    file_size_bytes: int
    sha256: str
    content_type: str | None = None


@dataclass(frozen=True, slots=True)
class ModelExecutionRecord:
    """Record of a model service execution — for provenance tracking."""
    run_id: RunId
    document_id: DocumentId
    version: Version
    task: InferenceTask
    provider: ModelProvider
    model_id: str
    input_hash: str | None = None       # Hash of input (for dedup/caching)
    output_summary: str | None = None    # Truncated output for quick reference
    confidence: float | None = None
    latency_ms: float | None = None
    token_usage: dict[str, int] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationRunRecord:
    """Record of a validation run — for provenance and audit."""
    run_id: RunId
    document_id: DocumentId
    version: Version
    template_id: TemplateId | None = None
    template_version: str | None = None  # e.g., "v1"
    status: ValidationStatus = ValidationStatus.UNKNOWN_DOCUMENT
    risk_score: float = 0.0
    is_authentic: bool = True
    violations: tuple[str, ...] = ()
    temporal_status: ExpiryStatus = ExpiryStatus.UNKNOWN
    temporal_tags: tuple[str, ...] = ()
    structural_confidence: float = 0.0
    model_execution_id: RunId | None = None  # FK to ModelExecutionRecord
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TemplateVersion:
    """Template with version tracking — for validator service."""
    template_id: TemplateId
    version: str                       # e.g., "1.0.0"
    document_type: str
    document_sub_type: str
    description: str
    field_count: int = 0
    temporal_rule_count: int = 0
    is_active: bool = True
    created_at: datetime | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Service API Contracts
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class IngestionRequest:
    """Unified ingestion request — used by Document service."""
    original_filename: str
    content: bytes
    content_type: str | None = None
    document_id: DocumentId | None = None  # If None, generate new
    privacy_hint: PrivacyClassification | None = None
    description_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Unified ingestion result — returned by Document service."""
    document_id: DocumentId
    version: Version
    processing_status: ProcessingStatus
    privacy: PrivacyClassification
    description: str | None
    metadata: dict[str, Any]
    storage_key: StorageKey
    sha256: str
    validation_status: ValidationStatus | None = None
    validation_temporal_status: ExpiryStatus | None = None
    validation_temporal_tags: tuple[str, ...] = ()
    chunk_count: int = 0


@dataclass(frozen=True, slots=True)
class ValidationRequest:
    """Unified validation request — used by Validator service."""
    document_id: DocumentId
    version: Version
    document_type: str | None = None
    document_sub_type: str | None = None
    extracted_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Unified validation result — returned by Validator service."""
    document_id: DocumentId
    version: Version
    status: ValidationStatus
    matched_template: TemplateId | None = None
    risk_score: float = 0.0
    is_authentic: bool = True
    temporal_status: ExpiryStatus = ExpiryStatus.UNKNOWN
    temporal_tags: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    alert_severity: AlertSeverity | None = None
    alert_type: str | None = None
    run_id: RunId | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Complete document metadata — used by all services for context."""
    document_id: DocumentId
    version: Version
    original_filename: str
    content_type: str | None
    file_kind: FileKind
    storage_key: StorageKey
    file_size_bytes: int
    sha256: str
    privacy: PrivacyClassification
    processing_status: ProcessingStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    extracted_text: str | None = None
    # Model extraction
    document_type: str | None = None
    document_sub_type: str | None = None
    extraction_confidence: float | None = None
    language_primary: str | None = None
    # Validation
    validation_status: ValidationStatus | None = None
    validation_temporal_status: ExpiryStatus | None = None
    validation_temporal_tags: tuple[str, ...] = ()
    validation_risk_score: float | None = None
    # Indexing
    chunk_count: int = 0
    semantic_index_status: str = "PENDING"
    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def make_chunk_id(document_id: DocumentId, version: Version, chunk_index: int) -> ChunkId:
    """Generate a deterministic chunk ID from document identity and index.

    This ensures the same chunk always gets the same ID, enabling
    bidirectional tracing between PostgreSQL and Qdrant.
    """
    import hashlib
    raw = f"{document_id}:{version}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_storage_key(document_id: DocumentId, version: Version, filename: str) -> StorageKey:
    """Generate a stable storage key for a document version."""
    safe_name = filename.replace("/", "_").replace("\\", "_")
    return f"documents/{document_id}/v{version}/{safe_name}"


def make_run_id() -> RunId:
    """Generate a unique run ID for validation/model execution tracking."""
    from uuid import uuid4
    return uuid4().hex
