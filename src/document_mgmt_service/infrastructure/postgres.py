from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentPrivacyClassification,
    DocumentProcessingStatus,
    DocumentSummaryRecord,
    DocumentVersionRecord,
    SemanticIndexStatus,
)
from document_mgmt_service.domain.ports import PostgreSQLDocumentRepository
from document_mgmt_service.schemas import apply_migrations


def create_psycopg_connection_factory(
    dsn: str,
    *,
    connect_timeout: int = 5,
    application_name: str = "document-mgmt-service",
) -> Callable[[], Any]:
    """Return a factory that opens a fresh psycopg connection per call.

    ``connect_timeout`` guards against a hanging TCP connect so the
    service can fail fast when the database container is not ready.
    ``application_name`` shows up in ``pg_stat_activity`` for easier
    debugging from inside the container.
    """
    def factory() -> Any:
        import psycopg  # type: ignore[import-not-found]

        return psycopg.connect(
            dsn,
            connect_timeout=connect_timeout,
            application_name=application_name,
        )

    return factory


class PostgreSQLRepositoryAdapter(PostgreSQLDocumentRepository):
    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

    def ping(self) -> None:
        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
        except Exception as exc:
            raise RuntimeError(
                f"PostgreSQL ping failed: {exc.__class__.__name__}: {exc}"
            ) from exc

    def ensure_schema(self) -> None:
        """Apply versioned migrations from the schema catalog."""
        apply_migrations(self._connection_factory)

    def next_version(self, document_id: str) -> int:
        with self._connection_factory() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM document_versions WHERE document_id = %s",
                    (document_id,),
                )
                row = cursor.fetchone()
                return int(row[0]) if row else 1

    def upsert_version(self, record: DocumentVersionRecord) -> None:
        with self._connection_factory() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO documents (document_id, latest_version, created_at, updated_at)
                    VALUES (%s, %s, COALESCE(%s, NOW()), COALESCE(%s, NOW()))
                    ON CONFLICT (document_id)
                    DO UPDATE SET
                        latest_version = GREATEST(documents.latest_version, EXCLUDED.latest_version),
                        updated_at = NOW()
                    """,
                    (
                        record.document_id,
                        record.version,
                        record.created_at,
                        record.updated_at,
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO document_versions (
                        document_id, version, original_filename, content_type, file_kind,
                        storage_key, file_size_bytes, sha256, privacy, processing_status,
                        metadata, description, extracted_text, extracted_text_excerpt,
                        semantic_index_status, semantic_indexed_at, chunk_count,
                        created_at, updated_at, completed_at, error_message,
                        model_extraction, description_safe, description_detailed,
                        extraction_confidence, document_type, document_sub_type,
                        language_primary, pii_types
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (document_id, version)
                    DO UPDATE SET
                        original_filename = EXCLUDED.original_filename,
                        content_type = EXCLUDED.content_type,
                        file_kind = EXCLUDED.file_kind,
                        storage_key = EXCLUDED.storage_key,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        sha256 = EXCLUDED.sha256,
                        privacy = EXCLUDED.privacy,
                        processing_status = EXCLUDED.processing_status,
                        metadata = EXCLUDED.metadata,
                        description = EXCLUDED.description,
                        extracted_text = EXCLUDED.extracted_text,
                        extracted_text_excerpt = EXCLUDED.extracted_text_excerpt,
                        semantic_index_status = EXCLUDED.semantic_index_status,
                        semantic_indexed_at = EXCLUDED.semantic_indexed_at,
                        chunk_count = EXCLUDED.chunk_count,
                        updated_at = EXCLUDED.updated_at,
                        completed_at = EXCLUDED.completed_at,
                        error_message = EXCLUDED.error_message,
                        model_extraction = EXCLUDED.model_extraction,
                        description_safe = EXCLUDED.description_safe,
                        description_detailed = EXCLUDED.description_detailed,
                        extraction_confidence = EXCLUDED.extraction_confidence,
                        document_type = EXCLUDED.document_type,
                        document_sub_type = EXCLUDED.document_sub_type,
                        language_primary = EXCLUDED.language_primary,
                        pii_types = EXCLUDED.pii_types
                    """,
                    (
                        record.document_id,
                        record.version,
                        record.original_filename,
                        record.content_type,
                        record.file_kind.value,
                        record.storage_key,
                        record.file_size_bytes,
                        record.sha256,
                        record.privacy.value,
                        record.processing_status.value,
                        json.dumps(record.metadata, default=str),
                        record.description,
                        record.extracted_text,
                        record.extracted_text_excerpt,
                        record.semantic_index_status.value,
                        record.semantic_indexed_at,
                        record.chunk_count,
                        record.created_at,
                        record.updated_at,
                        record.completed_at,
                        record.error_message,
                        json.dumps(record.model_extraction, default=str) if record.model_extraction else None,
                        record.description_safe,
                        record.description_detailed,
                        record.extraction_confidence,
                        record.document_type,
                        record.document_sub_type,
                        record.language_primary,
                        json.dumps(record.pii_types) if record.pii_types else None,
                    ),
                )
            conn.commit()

    def get_version(self, document_id: str, version: int) -> DocumentVersionRecord | None:
        with self._connection_factory() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, version, original_filename, content_type, file_kind,
                           storage_key, file_size_bytes, sha256, privacy, processing_status,
                           metadata, description, extracted_text, extracted_text_excerpt,
                           semantic_index_status, semantic_indexed_at, chunk_count,
                           created_at, updated_at, completed_at, error_message,
                           model_extraction, description_safe, description_detailed,
                           extraction_confidence, document_type, document_sub_type,
                           language_primary, pii_types,
                           summary, extracted_fields, owner_type, relation, relation_name, expiry_date
                    FROM document_versions
                    WHERE document_id = %s AND version = %s
                    """,
                    (document_id, version),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_versions(self, document_id: str) -> list[DocumentVersionRecord]:
        with self._connection_factory() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, version, original_filename, content_type, file_kind,
                           storage_key, file_size_bytes, sha256, privacy, processing_status,
                           metadata, description, extracted_text, extracted_text_excerpt,
                           semantic_index_status, semantic_indexed_at, chunk_count,
                           created_at, updated_at, completed_at, error_message,
                           model_extraction, description_safe, description_detailed,
                           extraction_confidence, document_type, document_sub_type,
                           language_primary, pii_types,
                           summary, extracted_fields, owner_type, relation, relation_name, expiry_date
                    FROM document_versions
                    WHERE document_id = %s
                    ORDER BY version ASC
                    """,
                    (document_id,),
                )
                rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_documents(self) -> list[DocumentSummaryRecord]:
        with self._connection_factory() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.document_id, d.latest_version, v.processing_status, v.privacy,
                           v.metadata, v.description, v.summary, v.owner_type, v.relation, v.relation_name, v.expiry_date,
                           v.extracted_fields, v.semantic_index_status, v.chunk_count,
                           d.created_at, d.updated_at
                    FROM documents d
                    JOIN document_versions v
                      ON v.document_id = d.document_id
                     AND v.version = d.latest_version
                    ORDER BY d.updated_at DESC, d.document_id ASC
                    """
                )
                rows = cursor.fetchall()
        return [self._row_to_summary(row) for row in rows]

    def close(self) -> None:
        return None

    def _row_to_record(self, row: Any) -> DocumentVersionRecord:
        return DocumentVersionRecord(
            document_id=row[0],
            version=int(row[1]),
            original_filename=row[2],
            content_type=row[3],
            file_kind=DocumentFileKind(row[4]),
            storage_key=row[5],
            file_size_bytes=int(row[6]),
            sha256=row[7],
            privacy=DocumentPrivacyClassification(row[8]),
            processing_status=DocumentProcessingStatus(row[9]),
            metadata=dict(row[10] or {}),
            description=row[11],
            extracted_text=row[12],
            extracted_text_excerpt=row[13],
            semantic_index_status=SemanticIndexStatus(row[14]),
            semantic_indexed_at=row[15],
            chunk_count=int(row[16] or 0),
            created_at=row[17],
            updated_at=row[18],
            completed_at=row[19],
            error_message=row[20],
            model_extraction=dict(row[21]) if row[21] else None,
            description_safe=row[22],
            description_detailed=row[23],
            extraction_confidence=float(row[24]) if row[24] is not None else None,
            document_type=row[25],
            document_sub_type=row[26],
            language_primary=row[27],
            pii_types=list(row[28]) if row[28] else None,
            summary=row[29],
            extracted_fields=dict(row[30]) if row[30] else None,
            owner_type=row[31],
            relation=row[32],
            relation_name=row[33],
            expiry_date=row[34],
        )

    def _row_to_summary(self, row: Any) -> DocumentSummaryRecord:
        # Columns from list_documents SELECT, in order:
        #  0 d.document_id    1 latest_version        2 processing_status
        #  3 privacy          4 metadata              5 description
        #  6 summary          7 owner_type            8 relation
        #  9 relation_name    10 expiry_date          11 extracted_fields
        #  12 semantic_index_status   13 chunk_count  14 created_at
        #  15 updated_at
        return DocumentSummaryRecord(
            document_id=row[0],
            latest_version=int(row[1]),
            processing_status=DocumentProcessingStatus(row[2]),
            privacy=DocumentPrivacyClassification(row[3]),
            metadata=dict(row[4] or {}),
            description=row[5],
            summary=row[6],
            owner_type=row[7],
            relation=row[8],
            relation_name=row[9],
            expiry_date=row[10],
            extracted_fields=dict(row[11]) if row[11] else None,
            semantic_index_status=SemanticIndexStatus(row[12]),
            chunk_count=int(row[13] or 0),
            created_at=row[14],
            updated_at=row[15],
        )
