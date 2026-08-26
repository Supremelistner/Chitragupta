"""Explicit Qdrant payload schema for document chunk vectors.

Qdrant is NOT the canonical document store or authorization authority.
It holds only what is needed for semantic retrieval:

    - chunk vectors (embeddings)
    - payload fields for filtering and provenance

The Qdrant point ID uses the chunk_id (a stable, deterministic string).
This allows bidirectional tracing: PostgreSQL chunk → Qdrant point via
chunk_id, and Qdrant point → PostgreSQL record via document_id + version.

Ownership:
    Qdrant     = vectors + retrieval payload (this module defines the contract)
    PostgreSQL = canonical metadata (see schemas/__init__.py)
    Filesystem = document binaries (see infrastructure/storage.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentPrivacyClassification,
    SemanticIndexStatus,
)


@dataclass(frozen=True, slots=True)
class QdrantChunkPayload:
    """Typed representation of the payload stored alongside each Qdrant vector.

    Every field here is indexed or used for filtering/retrieval.  The schema
    is intentionally flat to keep Qdrant payload queries simple.

    Fields:
        chunk_id:     Stable, deterministic ID — also the Qdrant point ID.
        document_id:  FK back to PostgreSQL documents table.
        version:      FK back to PostgreSQL document_versions.
        chunk_index:  Position within the document's chunk sequence.
        page_number:  1-based page number from OCR extraction (null if unknown).
        char_start:   Start character offset in the extracted text.
        char_end:     End character offset in the extracted text.
        text:         The actual chunk text (stored for payload-only retrieval).
        privacy:      Privacy classification — used for filtered search.
        file_kind:    Original file type (PDF, IMAGE, etc.).
        original_filename:  Human-readable filename for provenance.
        content_type:       MIME type of the original file.
        description:        Safe, non-identifying description.
        semantic_index_status: Current indexing state.
        retrieval_metadata:    Extensible dict for future filtering (e.g., entities, tags).
    """

    chunk_id: str
    document_id: str
    version: int
    chunk_index: int
    page_number: int | None
    text: str
    char_start: int
    char_end: int
    privacy: str  # DocumentPrivacyClassification value
    file_kind: str  # DocumentFileKind value
    original_filename: str
    content_type: str | None
    description: str | None
    semantic_index_status: str  # SemanticIndexStatus value
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    # Model extraction fields
    document_type: str | None = None
    document_sub_type: str | None = None
    language_primary: str | None = None
    extraction_confidence: float | None = None
    pii_types: tuple[str, ...] = ()
    description_safe: str | None = None

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize to a flat dict suitable for Qdrant point payload."""
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "version": self.version,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "privacy": self.privacy,
            "file_kind": self.file_kind,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "description": self.description,
            "semantic_index_status": self.semantic_index_status,
        }
        if self.page_number is not None:
            payload["page_number"] = self.page_number
        if self.retrieval_metadata:
            payload["retrieval_metadata"] = self.retrieval_metadata
        # Model extraction fields
        if self.document_type is not None:
            payload["document_type"] = self.document_type
        if self.document_sub_type is not None:
            payload["document_sub_type"] = self.document_sub_type
        if self.language_primary is not None:
            payload["language_primary"] = self.language_primary
        if self.extraction_confidence is not None:
            payload["extraction_confidence"] = self.extraction_confidence
        if self.pii_types:
            payload["pii_types"] = list(self.pii_types)
        if self.description_safe is not None:
            payload["description_safe"] = self.description_safe
        return payload

    @classmethod
    def from_chunk_record(cls, record: Any) -> QdrantChunkPayload:  # noqa: ANN401
        """Build a QdrantChunkRecord from a domain SemanticChunkRecord."""
        return cls(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            version=record.version,
            chunk_index=record.chunk_index,
            page_number=record.page_number,
            text=record.text,
            char_start=record.start_char,
            char_end=record.end_char,
            privacy=record.privacy.value,
            file_kind=record.metadata.get("file_kind", DocumentFileKind.UNKNOWN.value),
            original_filename=record.original_filename or "",
            content_type=record.content_type,
            description=record.description,
            semantic_index_status=record.metadata.get(
                "semantic_index_status", SemanticIndexStatus.PENDING.value
            ),
            retrieval_metadata={
                k: v
                for k, v in record.metadata.items()
                if k not in {"file_kind", "semantic_index_status"}
            },
            document_type=record.metadata.get("document_type"),
            document_sub_type=record.metadata.get("document_sub_type"),
            language_primary=record.metadata.get("language_primary"),
            extraction_confidence=record.metadata.get("extraction_confidence"),
            pii_types=tuple(record.metadata.get("pii_types", [])),
            description_safe=record.metadata.get("description_safe"),
        )


# Qdrant collection configuration for V1
QDRANT_COLLECTION_CONFIG = {
    "name": "document_chunks",
    "description": "Semantic retrieval index for document chunks.",
    "vector_dimension": "configured via DOCUMENT_SERVICE_SEMANTIC_EMBED_DIM (default 256)",
    "distance_metric": "Cosine",
    "indexed_payload_fields": [
        "document_id",
        "version",
        "chunk_index",
        "page_number",
        "privacy",
        "file_kind",
        "semantic_index_status",
        "document_type",
        "document_sub_type",
        "language_primary",
        "pii_types",
    ],
    "point_id_strategy": "chunk_id (deterministic string)",
    "notes": [
        "Qdrant is NOT the canonical store — use PostgreSQL for authoritative state.",
        "Payload text is stored for convenience; source of truth is in PostgreSQL.",
        "Privacy field enables filtered vector search without round-tripping to Postgres.",
        "point_id = chunk_id ensures bidirectional traceability.",
    ],
}
