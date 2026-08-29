from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DocumentPrivacyClassification(str, Enum):
    OPEN = "OPEN"
    OPEN_NOT_PUBLIC = "OPEN_NOT_PUBLIC"
    PRIVATE = "PRIVATE"
    SENSITIVE = "SENSITIVE"


class DocumentProcessingStatus(str, Enum):
    RECEIVED = "RECEIVED"
    STORED = "STORED"
    OCR_IN_PROGRESS = "OCR_IN_PROGRESS"
    OCR_COMPLETE = "OCR_COMPLETE"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    METADATA_GENERATED = "METADATA_GENERATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SemanticIndexStatus(str, Enum):
    PENDING = "PENDING"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class DocumentFileKind(str, Enum):
    PDF = "PDF"
    IMAGE = "IMAGE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DocumentVersionRecord:
    document_id: str
    version: int
    original_filename: str
    content_type: str | None
    file_kind: DocumentFileKind
    storage_key: str
    file_size_bytes: int
    sha256: str
    privacy: DocumentPrivacyClassification
    processing_status: DocumentProcessingStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    extracted_text: str | None = None
    extracted_text_excerpt: str | None = None
    semantic_index_status: SemanticIndexStatus = SemanticIndexStatus.PENDING
    semantic_indexed_at: datetime | None = None
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    # Model extraction fields (populated by model service)
    model_extraction: dict[str, Any] | None = None
    description_safe: str | None = None
    description_detailed: str | None = None
    extraction_confidence: float | None = None
    document_type: str | None = None
    document_sub_type: str | None = None
    language_primary: str | None = None
    pii_types: list[str] | None = None
    # V2: owner / relation extracted by the vision model
    owner_type: str | None = None         # SELF | MOTHER | FATHER | SPOUSE | CHILD | OTHER
    relation: str | None = None           # e.g. "Mother"
    relation_name: str | None = None      # e.g. "Priya Sharma"
    # V2: split summary (safe) from fields (PII, encrypted)
    summary: str | None = None
    extracted_fields: dict[str, Any] | None = None
    # V2: extracted expiry date if visible on the document
    expiry_date: datetime | None = None


@dataclass(frozen=True, slots=True)
class DocumentIngestionRequest:
    original_filename: str
    content: bytes
    content_type: str | None = None
    document_id: str | None = None
    privacy_hint: DocumentPrivacyClassification | None = None
    description_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentIngestionResult:
    document_id: str
    version: int
    processing_status: DocumentProcessingStatus
    privacy: DocumentPrivacyClassification
    description: str | None
    metadata: dict[str, Any]
    storage_key: str
    sha256: str
    semantic_index_status: SemanticIndexStatus
    chunk_count: int


@dataclass(frozen=True, slots=True)
class DocumentStatusResponse:
    document_id: str
    version: int
    processing_status: DocumentProcessingStatus
    privacy: DocumentPrivacyClassification
    metadata: dict[str, Any]
    description: str | None
    semantic_index_status: SemanticIndexStatus
    chunk_count: int
    error_message: str | None
    created_at: datetime | None
    updated_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class DocumentSummaryRecord:
    document_id: str
    latest_version: int
    processing_status: DocumentProcessingStatus
    privacy: DocumentPrivacyClassification
    metadata: dict[str, Any]
    description: str | None
    summary: str | None = None
    owner_type: str | None = None
    relation: str | None = None
    relation_name: str | None = None
    expiry_date: datetime | None = None
    extracted_fields: dict[str, Any] | None = None
    semantic_index_status: SemanticIndexStatus = SemanticIndexStatus.PENDING
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SemanticChunkRecord:
    chunk_id: str
    document_id: str
    version: int
    chunk_index: int
    page_number: int | None
    text: str
    start_char: int
    end_char: int
    privacy: DocumentPrivacyClassification
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SemanticDocumentSearchResult:
    document_id: str
    version: int
    score: float
    privacy: DocumentPrivacyClassification
    description: str | None
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SemanticContentSearchResult:
    chunk_id: str
    document_id: str
    version: int
    score: float
    privacy: DocumentPrivacyClassification
    text: str
    metadata: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SemanticEvidenceResult:
    document_id: str
    version: int
    query: str
    score: float
    privacy: DocumentPrivacyClassification
    text: str
    metadata: dict[str, Any]
    provenance: dict[str, Any]
