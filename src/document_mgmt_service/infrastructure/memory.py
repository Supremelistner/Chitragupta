from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from document_mgmt_service.domain.models import DocumentSummaryRecord, DocumentVersionRecord
from document_mgmt_service.domain.ports import PostgreSQLDocumentRepository


class InMemoryDocumentRepository(PostgreSQLDocumentRepository):
    def __init__(self) -> None:
        self._records: dict[tuple[str, int], DocumentVersionRecord] = {}

    def ping(self) -> None:
        return None

    def ensure_schema(self) -> None:
        return None

    def next_version(self, document_id: str) -> int:
        versions = [version for (doc_id, version) in self._records if doc_id == document_id]
        return (max(versions) + 1) if versions else 1

    def upsert_version(self, record: DocumentVersionRecord) -> None:
        self._records[(record.document_id, record.version)] = deepcopy(record)

    def get_version(self, document_id: str, version: int) -> DocumentVersionRecord | None:
        record = self._records.get((document_id, version))
        return deepcopy(record) if record is not None else None

    def list_versions(self, document_id: str) -> list[DocumentVersionRecord]:
        versions = [
            deepcopy(record)
            for (doc_id, _), record in self._records.items()
            if doc_id == document_id
        ]
        versions.sort(key=lambda item: item.version)
        return versions

    def list_documents(self) -> list[DocumentSummaryRecord]:
        latest_by_doc: dict[str, DocumentVersionRecord] = {}
        for (document_id, _), record in self._records.items():
            current = latest_by_doc.get(document_id)
            if current is None or record.version > current.version:
                latest_by_doc[document_id] = record
        summaries = [
            DocumentSummaryRecord(
                document_id=record.document_id,
                latest_version=record.version,
                processing_status=record.processing_status,
                privacy=record.privacy,
                metadata=deepcopy(record.metadata),
                description=record.description,
                semantic_index_status=record.semantic_index_status,
                chunk_count=record.chunk_count,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            for record in latest_by_doc.values()
        ]
        summaries.sort(key=lambda item: item.document_id)
        return summaries

    def close(self) -> None:
        return None
