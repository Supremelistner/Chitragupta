"""SQLite-backed document repository.

Persistent replacement for InMemoryDocumentRepository.
Uses only Python stdlib sqlite3 — zero external dependencies.
Auto-creates schema on first use.

Can be swapped for PostgreSQL in production without changing domain logic.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    latest_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS document_versions (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    original_filename TEXT,
    content_type TEXT,
    file_kind TEXT,
    storage_key TEXT,
    file_size_bytes INTEGER DEFAULT 0,
    sha256 TEXT,
    privacy TEXT DEFAULT 'OPEN_NOT_PUBLIC',
    processing_status TEXT DEFAULT 'RECEIVED',
    metadata TEXT DEFAULT '{}',
    description TEXT,
    extracted_text TEXT,
    extracted_text_excerpt TEXT,
    semantic_index_status TEXT DEFAULT 'PENDING',
    semantic_indexed_at TEXT,
    chunk_count INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    model_extraction TEXT,
    description_safe TEXT,
    description_detailed TEXT,
    extraction_confidence REAL,
    document_type TEXT,
    document_sub_type TEXT,
    language_primary TEXT,
    pii_types TEXT,
    PRIMARY KEY (document_id, version)
);
"""


class SQLiteDocumentRepository(PostgreSQLDocumentRepository):
    """SQLite-backed document repository.

    File-based persistence: all document metadata survives service restarts.
    Uses WAL mode for concurrent read performance.
    """

    def __init__(self, db_path: str = "./data/chitragupta.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def ping(self) -> None:
        if self._conn is None:
            raise RuntimeError("SQLite connection is closed")
        self._conn.execute("SELECT 1")

    def ensure_schema(self) -> None:
        if self._conn is None:
            self._connect()

    def next_version(self, document_id: str) -> int:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM document_versions WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return int(row[0]) if row else 1

    def upsert_version(self, record: DocumentVersionRecord) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO documents (document_id, latest_version, created_at, updated_at)
            VALUES (?, ?, COALESCE(?, ?), ?)
            ON CONFLICT (document_id) DO UPDATE SET
                latest_version = MAX(documents.latest_version, excluded.latest_version),
                updated_at = excluded.updated_at
            """,
            (
                record.document_id,
                record.version,
                record.created_at.isoformat() if record.created_at else None,
                now,
                now,
            ),
        )

        self._conn.execute(
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
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT (document_id, version) DO UPDATE SET
                original_filename = excluded.original_filename,
                content_type = excluded.content_type,
                file_kind = excluded.file_kind,
                storage_key = excluded.storage_key,
                file_size_bytes = excluded.file_size_bytes,
                sha256 = excluded.sha256,
                privacy = excluded.privacy,
                processing_status = excluded.processing_status,
                metadata = excluded.metadata,
                description = excluded.description,
                extracted_text = excluded.extracted_text,
                extracted_text_excerpt = excluded.extracted_text_excerpt,
                semantic_index_status = excluded.semantic_index_status,
                semantic_indexed_at = excluded.semantic_indexed_at,
                chunk_count = excluded.chunk_count,
                updated_at = excluded.updated_at,
                completed_at = excluded.completed_at,
                error_message = excluded.error_message,
                model_extraction = excluded.model_extraction,
                description_safe = excluded.description_safe,
                description_detailed = excluded.description_detailed,
                extraction_confidence = excluded.extraction_confidence,
                document_type = excluded.document_type,
                document_sub_type = excluded.document_sub_type,
                language_primary = excluded.language_primary,
                pii_types = excluded.pii_types
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
                record.semantic_indexed_at.isoformat() if record.semantic_indexed_at else None,
                record.chunk_count,
                record.created_at.isoformat() if record.created_at else None,
                record.updated_at.isoformat() if record.updated_at else None,
                record.completed_at.isoformat() if record.completed_at else None,
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
        self._conn.commit()

    def get_version(self, document_id: str, version: int) -> DocumentVersionRecord | None:
        assert self._conn is not None
        row = self._conn.execute(
            """
            SELECT document_id, version, original_filename, content_type, file_kind,
                   storage_key, file_size_bytes, sha256, privacy, processing_status,
                   metadata, description, extracted_text, extracted_text_excerpt,
                   semantic_index_status, semantic_indexed_at, chunk_count,
                   created_at, updated_at, completed_at, error_message,
                   model_extraction, description_safe, description_detailed,
                   extraction_confidence, document_type, document_sub_type,
                   language_primary, pii_types
            FROM document_versions
            WHERE document_id = ? AND version = ?
            """,
            (document_id, version),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_versions(self, document_id: str) -> list[DocumentVersionRecord]:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT document_id, version, original_filename, content_type, file_kind,
                   storage_key, file_size_bytes, sha256, privacy, processing_status,
                   metadata, description, extracted_text, extracted_text_excerpt,
                   semantic_index_status, semantic_indexed_at, chunk_count,
                   created_at, updated_at, completed_at, error_message,
                   model_extraction, description_safe, description_detailed,
                   extraction_confidence, document_type, document_sub_type,
                   language_primary, pii_types
            FROM document_versions
            WHERE document_id = ?
            ORDER BY version ASC
            """,
            (document_id,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_documents(self) -> list[DocumentSummaryRecord]:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT d.document_id, d.latest_version, v.processing_status, v.privacy,
                   v.metadata, v.description, v.semantic_index_status, v.chunk_count,
                   d.created_at, d.updated_at
            FROM documents d
            JOIN document_versions v
              ON v.document_id = d.document_id
             AND v.version = d.latest_version
            ORDER BY d.updated_at DESC, d.document_id ASC
            """
        ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Row mapping --

    def _row_to_record(self, row: Any) -> DocumentVersionRecord:
        return DocumentVersionRecord(
            document_id=row["document_id"],
            version=int(row["version"]),
            original_filename=row["original_filename"],
            content_type=row["content_type"],
            file_kind=DocumentFileKind(row["file_kind"]),
            storage_key=row["storage_key"],
            file_size_bytes=int(row["file_size_bytes"] or 0),
            sha256=row["sha256"],
            privacy=DocumentPrivacyClassification(row["privacy"]),
            processing_status=DocumentProcessingStatus(row["processing_status"]),
            metadata=json.loads(row["metadata"] or "{}"),
            description=row["description"],
            extracted_text=row["extracted_text"],
            extracted_text_excerpt=row["extracted_text_excerpt"],
            semantic_index_status=SemanticIndexStatus(row["semantic_index_status"]),
            semantic_indexed_at=_parse_iso(row["semantic_indexed_at"]),
            chunk_count=int(row["chunk_count"] or 0),
            created_at=_parse_iso(row["created_at"]),
            updated_at=_parse_iso(row["updated_at"]),
            completed_at=_parse_iso(row["completed_at"]),
            error_message=row["error_message"],
            model_extraction=json.loads(row["model_extraction"]) if row["model_extraction"] else None,
            description_safe=row["description_safe"],
            description_detailed=row["description_detailed"],
            extraction_confidence=float(row["extraction_confidence"]) if row["extraction_confidence"] is not None else None,
            document_type=row["document_type"],
            document_sub_type=row["document_sub_type"],
            language_primary=row["language_primary"],
            pii_types=json.loads(row["pii_types"]) if row["pii_types"] else None,
        )

    def _row_to_summary(self, row: Any) -> DocumentSummaryRecord:
        return DocumentSummaryRecord(
            document_id=row["document_id"],
            latest_version=int(row["latest_version"]),
            processing_status=DocumentProcessingStatus(row["processing_status"]),
            privacy=DocumentPrivacyClassification(row["privacy"]),
            metadata=json.loads(row["metadata"] or "{}"),
            description=row["description"],
            semantic_index_status=SemanticIndexStatus(row["semantic_index_status"]),
            chunk_count=int(row["chunk_count"] or 0),
            created_at=_parse_iso(row["created_at"]),
            updated_at=_parse_iso(row["updated_at"]),
        )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
