from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import logging
from pathlib import Path
import tempfile
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger("document_mgmt_service.ingestion")

from document_mgmt_service.application.metadata import (
    classify_privacy,
    generate_safe_description,
    infer_file_kind,
)
from document_mgmt_service.application.search import SemanticSearchService
from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentIngestionRequest,
    DocumentIngestionResult,
    DocumentPrivacyClassification,
    DocumentProcessingStatus,
    DocumentStatusResponse,
    DocumentVersionRecord,
    SemanticIndexStatus,
)
from document_mgmt_service.domain.ports import FileStorage, OCRService, PostgreSQLDocumentRepository
from document_mgmt_service.infrastructure.image_utils import resize_image


class ModelProviderProtocol:
    """Protocol for model service text extraction and classification (optional dependency)."""
    def extract_text(self, image_bytes: bytes, mime_type: str) -> str: ...
    def classify_document(self, image_bytes: bytes, mime_type: str) -> dict[str, Any]: ...


class DocumentValidatorProtocol:
    """Protocol for document validation during ingestion."""
    def validate(
        self,
        document_id: str,
        version: int,
        metadata: dict[str, Any],
        document_type: str | None = None,
        document_sub_type: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class IngestionDependencies:
    repository: PostgreSQLDocumentRepository
    storage: FileStorage
    ocr: OCRService | None = None
    semantic_search: SemanticSearchService | None = None
    model_provider: ModelProviderProtocol | None = None
    validator: DocumentValidatorProtocol | None = None


class IngestionError(RuntimeError):
    pass


def _parse_privacy(classification: str | None) -> DocumentPrivacyClassification:
    """Map model's privacy classification string to our enum."""
    if not classification:
        return DocumentPrivacyClassification.OPEN_NOT_PUBLIC
    mapping = {
        "SENSITIVE": DocumentPrivacyClassification.SENSITIVE,
        "PRIVATE": DocumentPrivacyClassification.PRIVATE,
        "OPEN_NOT_PUBLIC": DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
        "OPEN": DocumentPrivacyClassification.OPEN,
    }
    return mapping.get(classification.upper(), DocumentPrivacyClassification.OPEN_NOT_PUBLIC)


class IngestionService:
    def __init__(
        self,
        *,
        dependencies: IngestionDependencies,
        uuid_factory: Callable[[], str] = lambda: uuid4().hex,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._dependencies = dependencies
        self._uuid_factory = uuid_factory
        self._clock = clock

    def ingest(self, request: DocumentIngestionRequest) -> DocumentIngestionResult:
        repository = self._dependencies.repository
        storage = self._dependencies.storage
        ocr = self._dependencies.ocr
        semantic_search = self._dependencies.semantic_search

        document_id = request.document_id or self._uuid_factory()
        version = repository.next_version(document_id)
        now = self._clock()
        file_kind = infer_file_kind(request.original_filename, request.content_type)
        checksum = sha256(request.content).hexdigest()

        if not request.content:
            raise IngestionError("Uploaded file is empty")

        temp_path: Path | None = None
        storage_key = self._storage_key(document_id, version, request.original_filename)

        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(request.content)
                temp_path = Path(temp_file.name)

            initial_record = DocumentVersionRecord(
                document_id=document_id,
                version=version,
                original_filename=request.original_filename,
                content_type=request.content_type,
                file_kind=file_kind,
                storage_key=storage_key,
                file_size_bytes=len(request.content),
                sha256=checksum,
                privacy=request.privacy_hint or DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
                processing_status=DocumentProcessingStatus.RECEIVED,
                metadata=dict(request.metadata),
                semantic_index_status=SemanticIndexStatus.PENDING,
                created_at=now,
                updated_at=now,
            )
            repository.upsert_version(initial_record)

            storage.put(temp_path, storage_key)
            stored_record = self._replace_status(
                initial_record,
                status=DocumentProcessingStatus.STORED,
                updated_at=now,
            )
            repository.upsert_version(stored_record)

            ocr_started = self._replace_status(
                stored_record,
                status=DocumentProcessingStatus.OCR_IN_PROGRESS,
                updated_at=now,
            )
            repository.upsert_version(ocr_started)

            extracted_text = ""
            classification: dict[str, Any] = {}
            if file_kind in {DocumentFileKind.PDF, DocumentFileKind.IMAGE}:
                model_provider = self._dependencies.model_provider
                if model_provider is not None:
                    try:
                        mime = request.content_type or (
                            "image/jpeg" if file_kind == DocumentFileKind.IMAGE else "application/pdf"
                        )
                        model_bytes = request.content
                        model_mime = mime
                        if file_kind == DocumentFileKind.IMAGE:
                            model_bytes, model_mime = resize_image(
                                request.content, mime,
                            )
                        # Text extraction
                        extracted_text = model_provider.extract_text(
                            image_bytes=model_bytes,
                            mime_type=model_mime,
                        )
                        # Classification — identify what the document is
                        try:
                            classification = model_provider.classify_document(
                                image_bytes=model_bytes,
                                mime_type=model_mime,
                            )
                            logger.info(
                                "Document %s classified as %s/%s (confidence=%.2f)",
                                document_id,
                                classification.get("document_type", "unknown"),
                                classification.get("document_sub_type", "unknown"),
                                classification.get("confidence", 0.0),
                            )
                        except Exception as cls_exc:
                            logger.warning("Classification failed: %s", cls_exc)
                    except Exception:
                        if ocr is not None:
                            extracted_text = ocr.extract_text(temp_path)
                elif ocr is not None:
                    extracted_text = ocr.extract_text(temp_path)

            ocr_complete = self._replace_status(
                ocr_started,
                status=DocumentProcessingStatus.OCR_COMPLETE,
                extracted_text=extracted_text or None,
                updated_at=now,
            )
            repository.upsert_version(ocr_complete)

            metadata = self._build_metadata(
                request=request,
                file_kind=file_kind,
                extracted_text=extracted_text,
                classification=classification,
            )

            # ── VALIDATION STEP ──────────────────────────────────────
            validator = self._dependencies.validator
            if validator is not None:
                try:
                    validation = validator.validate(
                        document_id=document_id,
                        version=version,
                        metadata=metadata,
                    )
                    # Merge validation results into metadata
                    validation_status = validation.get("validation_status")
                    metadata["validation_status"] = validation_status
                    metadata["validation_risk_score"] = validation.get("validation_risk_score", 0.0)
                    metadata["validation_matched_template"] = validation.get("validation_matched_template")
                    metadata["validation_violations"] = validation.get("validation_violations", [])
                    metadata["validation_temporal_status"] = validation.get("validation_temporal_status", "UNKNOWN")
                    metadata["validation_temporal_tags"] = validation.get("validation_temporal_tags", [])
                    metadata["validation_temporal_checks"] = validation.get("validation_temporal_checks", [])

                    # If validation flagged the document, store the alert
                    alert = validation.get("validation_alert")
                    if alert:
                        metadata["validation_alert"] = alert
                        logger.warning(
                            "Document %s v%d flagged by validator: status=%s risk=%.2f violations=%d temporal=%s",
                            document_id, version,
                            validation_status,
                            validation.get("validation_risk_score", 0.0),
                            len(validation.get("validation_violations", [])),
                            validation.get("validation_temporal_status"),
                        )
                except Exception as exc:
                    logger.warning("Validation failed for %s: %s", document_id, exc)
                    metadata["validation_status"] = "ERROR"
                    metadata["validation_error"] = str(exc)

            # Use model classification for description/privacy when available
            desc_from_model = classification.get("description", {})
            privacy_from_model = classification.get("privacy", {})

            description = (
                desc_from_model.get("safe")
                or desc_from_model.get("detailed")
                or generate_safe_description(
                    filename=request.original_filename,
                    file_kind=file_kind,
                    extracted_text=extracted_text,
                    metadata=metadata,
                    description_hint=request.description_hint,
                )
            )

            privacy = (
                _parse_privacy(privacy_from_model.get("classification"))
                if privacy_from_model.get("classification")
                else classify_privacy(
                    filename=request.original_filename,
                    content_type=request.content_type,
                    extracted_text=extracted_text,
                    description=description,
                    privacy_hint=request.privacy_hint,
                )
            )
            completed_record = self._replace_status(
                ocr_complete,
                status=DocumentProcessingStatus.COMPLETED,
                privacy=privacy,
                metadata=metadata,
                description=description,
                extracted_text=extracted_text or None,
                extracted_text_excerpt=(extracted_text[:500] if extracted_text else None),
                semantic_index_status=SemanticIndexStatus.INDEXING,
                completed_at=now,
                updated_at=now,
            )
            repository.upsert_version(completed_record)

            chunk_count = semantic_search.index_document(completed_record)
            indexed_record = self._replace_status(
                completed_record,
                status=DocumentProcessingStatus.INDEXED,
                semantic_index_status=SemanticIndexStatus.INDEXED,
                semantic_indexed_at=now,
                chunk_count=chunk_count,
                updated_at=now,
            )
            repository.upsert_version(indexed_record)

            return DocumentIngestionResult(
                document_id=document_id,
                version=version,
                processing_status=indexed_record.processing_status,
                privacy=indexed_record.privacy,
                description=indexed_record.description,
                metadata=indexed_record.metadata,
                storage_key=indexed_record.storage_key,
                sha256=indexed_record.sha256,
                semantic_index_status=indexed_record.semantic_index_status,
                chunk_count=indexed_record.chunk_count,
            )
        except Exception as exc:
            failed_record = DocumentVersionRecord(
                document_id=document_id,
                version=version,
                original_filename=request.original_filename,
                content_type=request.content_type,
                file_kind=file_kind,
                storage_key=storage_key,
                file_size_bytes=len(request.content),
                sha256=checksum,
                privacy=request.privacy_hint or DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
                processing_status=DocumentProcessingStatus.FAILED,
                metadata=dict(request.metadata),
                semantic_index_status=SemanticIndexStatus.FAILED,
                error_message=str(exc),
                created_at=now,
                updated_at=now,
            )
            try:
                repository.upsert_version(failed_record)
            finally:
                raise
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def get_status(self, document_id: str, version: int) -> DocumentStatusResponse:
        record = self._dependencies.repository.get_version(document_id, version)
        if record is None:
            raise IngestionError(f"Document {document_id} version {version} not found")
        return DocumentStatusResponse(
            document_id=record.document_id,
            version=record.version,
            processing_status=record.processing_status,
            privacy=record.privacy,
            metadata=record.metadata,
            description=record.description,
            semantic_index_status=record.semantic_index_status,
            chunk_count=record.chunk_count,
            error_message=record.error_message,
            created_at=record.created_at,
            updated_at=record.updated_at,
            completed_at=record.completed_at,
        )

    def update_model_extraction(
        self,
        document_id: str,
        version: int,
        extraction: dict[str, Any],
    ) -> DocumentVersionRecord:
        """Update a version record with model service extraction output.

        Populates the new model extraction columns from the structured
        JSON output of the model service.
        """
        record = self._dependencies.repository.get_version(document_id, version)
        if record is None:
            raise IngestionError(f"Document {document_id} version {version} not found")

        privacy_block = extraction.get("privacy", {})
        desc_block = extraction.get("description", {})
        lang_block = extraction.get("language", {})

        updated = self._replace_status(
            record,
            status=record.processing_status,
            model_extraction=extraction,
            description_safe=desc_block.get("safe"),
            description_detailed=desc_block.get("detailed"),
            extraction_confidence=extraction.get("extraction_confidence"),
            document_type=extraction.get("document_type"),
            document_sub_type=extraction.get("document_sub_type"),
            language_primary=lang_block.get("primary"),
            pii_types=privacy_block.get("pii_types"),
        )
        self._dependencies.repository.upsert_version(updated)
        return updated

    def _storage_key(self, document_id: str, version: int, filename: str) -> str:
        safe_name = filename.replace("/", "_").replace("\\", "_")
        return f"documents/{document_id}/v{version}/{safe_name}"

    def _replace_status(
        self,
        record: DocumentVersionRecord,
        *,
        status: DocumentProcessingStatus,
        privacy: DocumentPrivacyClassification | None = None,
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
        extracted_text: str | None = None,
        extracted_text_excerpt: str | None = None,
        semantic_index_status: SemanticIndexStatus | None = None,
        semantic_indexed_at: datetime | None = None,
        chunk_count: int | None = None,
        completed_at: datetime | None = None,
        updated_at: datetime | None = None,
        model_extraction: dict[str, Any] | None = None,
        description_safe: str | None = None,
        description_detailed: str | None = None,
        extraction_confidence: float | None = None,
        document_type: str | None = None,
        document_sub_type: str | None = None,
        language_primary: str | None = None,
        pii_types: list[str] | None = None,
    ) -> DocumentVersionRecord:
        return DocumentVersionRecord(
            document_id=record.document_id,
            version=record.version,
            original_filename=record.original_filename,
            content_type=record.content_type,
            file_kind=record.file_kind,
            storage_key=record.storage_key,
            file_size_bytes=record.file_size_bytes,
            sha256=record.sha256,
            privacy=privacy or record.privacy,
            processing_status=status,
            metadata=metadata if metadata is not None else record.metadata,
            description=description if description is not None else record.description,
            extracted_text=extracted_text if extracted_text is not None else record.extracted_text,
            extracted_text_excerpt=(
                extracted_text_excerpt
                if extracted_text_excerpt is not None
                else record.extracted_text_excerpt
            ),
            semantic_index_status=semantic_index_status or record.semantic_index_status,
            semantic_indexed_at=(
                semantic_indexed_at
                if semantic_indexed_at is not None
                else record.semantic_indexed_at
            ),
            chunk_count=chunk_count if chunk_count is not None else record.chunk_count,
            created_at=record.created_at,
            updated_at=updated_at or record.updated_at,
            completed_at=completed_at if completed_at is not None else record.completed_at,
            error_message=record.error_message,
            model_extraction=model_extraction if model_extraction is not None else record.model_extraction,
            description_safe=description_safe if description_safe is not None else record.description_safe,
            description_detailed=description_detailed if description_detailed is not None else record.description_detailed,
            extraction_confidence=extraction_confidence if extraction_confidence is not None else record.extraction_confidence,
            document_type=document_type if document_type is not None else record.document_type,
            document_sub_type=document_sub_type if document_sub_type is not None else record.document_sub_type,
            language_primary=language_primary if language_primary is not None else record.language_primary,
            pii_types=pii_types if pii_types is not None else record.pii_types,
        )

    def _build_metadata(
        self,
        *,
        request: DocumentIngestionRequest,
        file_kind: DocumentFileKind,
        extracted_text: str,
        classification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        words = [word for word in extracted_text.split() if word.strip()]
        base = {
            **dict(request.metadata),
            "filename": request.original_filename,
            "content_type": request.content_type,
            "file_kind": file_kind.value,
            "file_size_bytes": len(request.content),
            "text_character_count": len(extracted_text),
            "text_word_count": len(words),
        }
        # Merge model classification into metadata (replaces filename-based hints)
        if classification and not classification.get("error"):
            base["document_type"] = classification.get("document_type")
            base["document_sub_type"] = classification.get("document_sub_type")
            base["classification_confidence"] = classification.get("confidence")
            base["classification_indicators"] = classification.get("key_indicators", [])
            base["language_detected"] = classification.get("language_detected", [])
        return base
