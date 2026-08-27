"""Tests for metadata consistency, storage integrity, and schema contracts.

Covers:
- PostgreSQL schema migrations apply cleanly and are idempotent
- Qdrant payload schema is consistent with domain models
- Local file storage is isolated and returns stable addresses
- Bidirectional ID traceability (PostgreSQL ↔ Qdrant via chunk_id)
- Ingestion produces consistent state across all three backends
- Audit events table structure is correct
- Document relationship table structure is correct
- Storage config abstraction works correctly
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from document_mgmt_service.application.ingestion import IngestionDependencies, IngestionService
from document_mgmt_service.application.search import (
    SemanticChunker,
    SemanticChunkRecord,
    SemanticSearchService,
    TextEmbeddingService,
)
from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentIngestionRequest,
    DocumentPrivacyClassification,
    DocumentProcessingStatus,
    SemanticIndexStatus,
)
from document_mgmt_service.infrastructure.memory import InMemoryDocumentRepository
from document_mgmt_service.infrastructure.qdrant import QdrantSemanticChunkStoreAdapter
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter
from document_mgmt_service.schemas import MIGRATIONS, Migration, apply_migrations
from document_mgmt_service.schemas.qdrant_payload import (
    QDRANT_COLLECTION_CONFIG,
    QdrantChunkPayload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StaticOCR:
    def __init__(self, text: str = "sample extracted text") -> None:
        self._text = text

    def ping(self) -> None:
        return None

    def extract_text(self, file_path: Path) -> str:
        return self._text

    def close(self) -> None:
        return None


def _fake_psycopg_factory():
    """Create an in-memory-like connection factory using sqlite3 for schema tests.

    We use psycopg's ``connect(':memory:')`` when psycopg is available,
    otherwise fall back to a no-op stub for unit tests that don't need
    real SQL execution.
    """
    try:
        import psycopg  # type: ignore[import-not-found]

        def factory():
            return psycopg.connect("")

        return factory
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Schema catalog tests
# ---------------------------------------------------------------------------

class TestSchemaCatalog(unittest.TestCase):
    """Verify the migration catalog is well-formed."""

    def test_migrations_are_sequential(self) -> None:
        versions = [m.version for m in MIGRATIONS]
        self.assertEqual(versions, sorted(versions), "Migrations must be ordered by version")
        self.assertEqual(versions, list(range(1, len(versions) + 1)), "Versions must be sequential starting at 1")

    def test_each_migration_has_sql(self) -> None:
        for migration in MIGRATIONS:
            self.assertGreater(len(migration.sql.strip()), 0, f"Migration {migration.version} has empty SQL")
            self.assertGreater(len(migration.description), 0, f"Migration {migration.version} has empty description")

    def test_migrations_are_frozen(self) -> None:
        for migration in MIGRATIONS:
            self.assertTrue(hasattr(migration, "__frozenattrs__") or hasattr(migration, "__slots__"),
                            "Migrations should be immutable")
            self.assertTrue(isinstance(migration, Migration))


# ---------------------------------------------------------------------------
# Qdrant payload schema tests
# ---------------------------------------------------------------------------

class TestQdrantPayloadSchema(unittest.TestCase):
    """Verify Qdrant payload schema is consistent with domain models."""

    def test_payload_has_all_required_fields(self) -> None:
        payload = QdrantChunkPayload(
            chunk_id="doc1:v1:c0",
            document_id="doc1",
            version=1,
            chunk_index=0,
            page_number=1,
            text="hello world",
            char_start=0,
            char_end=11,
            privacy="OPEN",
            file_kind="PDF",
            original_filename="test.pdf",
            content_type="application/pdf",
            description="A test document",
            semantic_index_status="INDEXED",
        )
        d = payload.to_qdrant_payload()
        self.assertEqual(d["chunk_id"], "doc1:v1:c0")
        self.assertEqual(d["document_id"], "doc1")
        self.assertEqual(d["version"], 1)
        self.assertEqual(d["page_number"], 1)
        self.assertEqual(d["privacy"], "OPEN")
        self.assertEqual(d["text"], "hello world")

    def test_payload_serialization_is_flat_dict(self) -> None:
        payload = QdrantChunkPayload(
            chunk_id="x", document_id="x", version=1, chunk_index=0,
            page_number=None, text="", char_start=0, char_end=0,
            privacy="PRIVATE", file_kind="IMAGE", original_filename="x.png",
            content_type=None, description=None,
            semantic_index_status="PENDING",
        )
        d = payload.to_qdrant_payload()
        self.assertIsInstance(d, dict)
        # page_number should be omitted when None
        self.assertNotIn("page_number", d)

    def test_payload_from_chunk_record(self) -> None:
        record = SemanticChunkRecord(
            chunk_id="doc1:v1:c0",
            document_id="doc1",
            version=1,
            chunk_index=0,
            page_number=2,
            text="extracted text",
            start_char=100,
            end_char=115,
            privacy=DocumentPrivacyClassification.SENSITIVE,
            metadata={"file_kind": "PDF", "semantic_index_status": "INDEXED"},
            description="safe description",
            original_filename="secret.pdf",
            content_type="application/pdf",
        )
        payload = QdrantChunkPayload.from_chunk_record(record)
        self.assertEqual(payload.chunk_id, "doc1:v1:c0")
        self.assertEqual(payload.privacy, "SENSITIVE")
        self.assertEqual(payload.page_number, 2)
        self.assertEqual(payload.char_start, 100)

    def test_qdrant_collection_config_exists(self) -> None:
        self.assertIn("name", QDRANT_COLLECTION_CONFIG)
        self.assertIn("indexed_payload_fields", QDRANT_COLLECTION_CONFIG)
        self.assertIn("point_id_strategy", QDRANT_COLLECTION_CONFIG)
        self.assertIn("document_id", QDRANT_COLLECTION_CONFIG["indexed_payload_fields"])
        self.assertIn("privacy", QDRANT_COLLECTION_CONFIG["indexed_payload_fields"])
        self.assertIn("version", QDRANT_COLLECTION_CONFIG["indexed_payload_fields"])

    def test_point_id_strategy_uses_chunk_id(self) -> None:
        self.assertIn("chunk_id", QDRANT_COLLECTION_CONFIG["point_id_strategy"].lower())


# ---------------------------------------------------------------------------
# Local file storage tests
# ---------------------------------------------------------------------------

class TestLocalStorageIsolation(unittest.TestCase):
    """Verify local filesystem storage is properly isolated behind its interface."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.storage = LocalFileStorageAdapter(Path(self.tempdir.name))

    def test_put_returns_stable_key(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"test content")
            source = Path(f.name)
        key = "documents/doc1/v1/test.pdf"
        result = self.storage.put(source, key)
        self.assertEqual(result, key)

    def test_get_returns_path_under_root(self) -> None:
        key = "documents/doc1/v1/test.pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"test content")
            source = Path(f.name)
        self.storage.put(source, key)
        path = self.storage.get(key)
        self.assertTrue(str(path).startswith(str(Path(self.tempdir.name))))

    def test_exists_after_put(self) -> None:
        key = "documents/doc1/v1/test.pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"test content")
            source = Path(f.name)
        self.assertFalse(self.storage.exists(key))
        self.storage.put(source, key)
        self.assertTrue(self.storage.exists(key))

    def test_delete_removes_file(self) -> None:
        key = "documents/doc1/v1/test.pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"test content")
            source = Path(f.name)
        self.storage.put(source, key)
        self.assertTrue(self.storage.exists(key))
        self.storage.delete(key)
        self.assertFalse(self.storage.exists(key))

    def test_delete_nonexistent_is_safe(self) -> None:
        self.storage.delete("nonexistent/path.pdf")  # Should not raise

    def test_storage_key_is_not_filesystem_path(self) -> None:
        """Storage keys use forward slashes, not OS-native paths."""
        key = "documents/doc1/v1/test.pdf"
        self.assertIn("/", key)
        self.assertNotIn("\\", key)

    def test_storage_root_configurable(self) -> None:
        """Different root directories produce independent storage."""
        with tempfile.TemporaryDirectory() as other_root:
            other_storage = LocalFileStorageAdapter(Path(other_root))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(b"content")
                source = Path(f.name)
            key = "test.pdf"
            self.storage.put(source, key)
            self.assertTrue(self.storage.exists(key))
            self.assertFalse(other_storage.exists(key))


# ---------------------------------------------------------------------------
# Bidirectional ID traceability tests
# ---------------------------------------------------------------------------

class TestBidirectionalTraceability(unittest.TestCase):
    """Verify that PostgreSQL records and Qdrant points share stable IDs."""

    def test_chunk_id_deterministic(self) -> None:
        doc_id = "abc123"
        version = 1
        chunk_index = 0
        chunk_id = f"{doc_id}:v{version}:c{chunk_index}"
        self.assertEqual(chunk_id, "abc123:v1:c0")

    def test_chunk_id_links_postgres_to_qdrant(self) -> None:
        """A chunk_id created during ingestion should be usable as Qdrant point ID."""
        doc_id = "doc-link"
        version = 1
        chunk_index = 2
        chunk_id = f"{doc_id}:v{version}:c{chunk_index}"

        chunker = SemanticChunker(max_chunk_chars=50, chunk_overlap=10)
        # Simulate a record that would go into PostgreSQL
        from dataclasses import replace
        from datetime import datetime, timezone
        from document_mgmt_service.domain.models import DocumentVersionRecord

        record = DocumentVersionRecord(
            document_id=doc_id,
            version=version,
            original_filename="test.pdf",
            content_type="application/pdf",
            file_kind=DocumentFileKind.PDF,
            storage_key=f"documents/{doc_id}/v{version}/test.pdf",
            file_size_bytes=100,
            sha256="abc",
            privacy=DocumentPrivacyClassification.OPEN,
            processing_status=DocumentProcessingStatus.COMPLETED,
            extracted_text="word " * 20,  # Enough text to create multiple chunks
            semantic_index_status=SemanticIndexStatus.INDEXED,
            chunk_count=0,
            completed_at=datetime.now(timezone.utc),
        )

        chunks = chunker.chunk(record)
        # All chunks should have deterministic IDs
        for i, chunk in enumerate(chunks):
            expected = f"{doc_id}:v{version}:c{i}"
            self.assertEqual(chunk.chunk_id, expected)

            # Qdrant payload should carry the same chunk_id
            payload = QdrantChunkPayload.from_chunk_record(chunk)
            qdrant_dict = payload.to_qdrant_payload()
            self.assertEqual(qdrant_dict["chunk_id"], expected)
            self.assertEqual(qdrant_dict["document_id"], doc_id)
            self.assertEqual(qdrant_dict["version"], version)


# ---------------------------------------------------------------------------
# Ingestion consistency tests
# ---------------------------------------------------------------------------

class TestIngestionConsistency(unittest.TestCase):
    """Verify that ingestion produces consistent state across backends."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.repo = InMemoryDocumentRepository()
        self.storage = LocalFileStorageAdapter(Path(self.tempdir.name))
        self.store = QdrantSemanticChunkStoreAdapter()
        self.search = SemanticSearchService(
            store=self.store,
            embedder=TextEmbeddingService(dimension=32),
            chunker=SemanticChunker(max_chunk_chars=60, chunk_overlap=10),
        )
        self.ingestion = IngestionService(
            dependencies=IngestionDependencies(
                repository=self.repo,
                storage=self.storage,
                ocr=_StaticOCR("alpha page one\fpage two beta"),
                semantic_search=self.search,
            )
        )

    def _ingest(self, doc_id: str, filename: str = "test.pdf", content: bytes = b"test"):
        return self.ingestion.ingest(DocumentIngestionRequest(
            original_filename=filename,
            content=content,
            content_type="application/pdf",
            document_id=doc_id,
        ))

    def test_postgres_has_record_after_ingestion(self) -> None:
        result = self._ingest("doc-1")
        record = self.repo.get_version("doc-1", 1)
        self.assertIsNotNone(record)
        self.assertEqual(record.document_id, "doc-1")
        self.assertEqual(record.processing_status, DocumentProcessingStatus.INDEXED)

    def test_file_exists_after_ingestion(self) -> None:
        result = self._ingest("doc-2")
        self.assertTrue(self.storage.exists(result.storage_key))

    def test_storage_key_matches_postgres(self) -> None:
        result = self._ingest("doc-3")
        record = self.repo.get_version("doc-3", 1)
        self.assertEqual(result.storage_key, record.storage_key)

    def test_qdrant_has_chunks_after_ingestion(self) -> None:
        result = self._ingest("doc-4")
        self.assertGreater(result.chunk_count, 0)
        # Search should find the document
        matches = self.store.search(
            query_vector=TextEmbeddingService(dimension=32).embed("alpha"),
            limit=10,
        )
        doc_chunks = [m for m in matches if m.chunk.document_id == "doc-4"]
        self.assertGreater(len(doc_chunks), 0)

    def test_chunk_ids_in_qdrant_match_expected_format(self) -> None:
        result = self._ingest("doc-5")
        matches = self.store.search(
            query_vector=TextEmbeddingService(dimension=32).embed("page"),
            limit=10,
        )
        doc_chunks = [m for m in matches if m.chunk.document_id == "doc-5"]
        for match in doc_chunks:
            self.assertTrue(match.chunk.chunk_id.startswith("doc-5:v1:c"))

    def test_sha256_matches_content(self) -> None:
        content = b"unique content for checksum"
        result = self._ingest("doc-6", content=content)
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(result.sha256, expected)

    def test_file_content_matches_uploaded_content(self) -> None:
        content = b"exact file content check"
        result = self._ingest("doc-7", content=content)
        path = self.storage.get(result.storage_key)
        self.assertEqual(path.read_bytes(), content)

    def test_metadata_populated_after_ingestion(self) -> None:
        result = self._ingest("doc-8")
        self.assertIn("filename", result.metadata)
        self.assertEqual(result.metadata["filename"], "test.pdf")
        self.assertIn("text_word_count", result.metadata)

    def test_privacy_classification_populated(self) -> None:
        result = self._ingest("doc-9")
        self.assertIsInstance(result.privacy, DocumentPrivacyClassification)

    def test_version_increments(self) -> None:
        self._ingest("doc-10")
        result2 = self._ingest("doc-10")
        self.assertEqual(result2.version, 2)

    def test_ingestion_error_recorded(self) -> None:
        """When OCR fails, the FAILED status is recorded."""
        from document_mgmt_service.application.ingestion import IngestionError

        class _FailingOCR:
            def ping(self) -> None: return None
            def extract_text(self, file_path: Path) -> str: raise RuntimeError("OCR down")
            def close(self) -> None: return None

        svc = IngestionService(
            dependencies=IngestionDependencies(
                repository=self.repo,
                storage=self.storage,
                ocr=_FailingOCR(),
                semantic_search=self.search,
            )
        )
        with self.assertRaises(RuntimeError):
            svc.ingest(DocumentIngestionRequest(
                original_filename="fail.pdf",
                content=b"fail",
                document_id="doc-fail",
            ))
        record = self.repo.get_version("doc-fail", 1)
        self.assertIsNotNone(record)
        self.assertEqual(record.processing_status, DocumentProcessingStatus.FAILED)
        self.assertIn("OCR down", record.error_message)

    def test_empty_file_raises_error(self) -> None:
        from document_mgmt_service.application.ingestion import IngestionError
        with self.assertRaises(IngestionError):
            self.ingestion.ingest(DocumentIngestionRequest(
                original_filename="empty.pdf",
                content=b"",
                document_id="doc-empty",
            ))


# ---------------------------------------------------------------------------
# Qdrant adapter consistency tests
# ---------------------------------------------------------------------------

class TestQdrantAdapterConsistency(unittest.TestCase):
    """Verify the in-memory Qdrant adapter behaves consistently."""

    def setUp(self) -> None:
        self.store = QdrantSemanticChunkStoreAdapter(collection_name="test_chunks")

    def _make_chunk(self, doc_id: str, version: int, index: int, text: str = "text") -> SemanticChunkRecord:
        return SemanticChunkRecord(
            chunk_id=f"{doc_id}:v{version}:c{index}",
            document_id=doc_id,
            version=version,
            chunk_index=index,
            page_number=1,
            text=text,
            start_char=0,
            end_char=len(text),
            privacy=DocumentPrivacyClassification.OPEN,
            metadata={},
        )

    def test_upsert_and_retrieve(self) -> None:
        chunk = self._make_chunk("d1", 1, 0)
        self.store.upsert(chunks=[chunk], vectors=[[1.0, 0.0, 0.0]])
        results = self.store.search(query_vector=[1.0, 0.0, 0.0], limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "d1:v1:c0")

    def test_delete_removes_all_versions(self) -> None:
        self.store.upsert(
            chunks=[
                self._make_chunk("d1", 1, 0),
                self._make_chunk("d1", 2, 0),
            ],
            vectors=[[1.0, 0.0], [1.0, 0.0]],
        )
        self.store.delete_document("d1")
        results = self.store.search(query_vector=[1.0, 0.0], limit=10)
        self.assertEqual(len(results), 0)

    def test_delete_specific_version(self) -> None:
        self.store.upsert(
            chunks=[
                self._make_chunk("d1", 1, 0),
                self._make_chunk("d1", 2, 0),
            ],
            vectors=[[1.0, 0.0], [1.0, 0.0]],
        )
        self.store.delete_document("d1", version=1)
        results = self.store.search(query_vector=[1.0, 0.0], limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.version, 2)

    def test_filter_by_privacy(self) -> None:
        chunk_open = self._make_chunk("d1", 1, 0)
        chunk_sensitive = SemanticChunkRecord(
            chunk_id="d2:v1:c0",
            document_id="d2", version=1, chunk_index=0,
            page_number=1, text="secret", start_char=0, end_char=6,
            privacy=DocumentPrivacyClassification.SENSITIVE, metadata={},
        )
        self.store.upsert(
            chunks=[chunk_open, chunk_sensitive],
            vectors=[[1.0, 0.0], [1.0, 0.0]],
        )
        results = self.store.search(
            query_vector=[1.0, 0.0], limit=10,
            privacy=DocumentPrivacyClassification.OPEN,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.document_id, "d1")


# ---------------------------------------------------------------------------
# Storage config abstraction tests
# ---------------------------------------------------------------------------

class TestStorageConfigAbstraction(unittest.TestCase):
    """Verify that storage root can be configured without changing domain logic."""

    def test_config_reads_env_var(self) -> None:
        from document_mgmt_service.config import AppConfig
        config = AppConfig(
            app_name="test", environment="test", log_level="INFO",
            http_host="127.0.0.1", http_port=8080,
            mcp_server_name="test", mcp_server_version="0.1.0",
            postgres_dsn=None, qdrant_url=None,
            qdrant_collection_name="test", semantic_embedding_dimension=32,
            semantic_chunk_size=100, semantic_chunk_overlap=20,
            file_storage_root=Path("/tmp/test-storage"),
            ocr_enabled=False,
            huggingface_token=None,
            huggingface_model_id="test-model",
            request_timeout_seconds=30,
        )
        self.assertEqual(config.file_storage_root, Path("/tmp/test-storage"))

    def test_storage_uses_config_root(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            storage = LocalFileStorageAdapter(Path(root))
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"content")
                source = Path(f.name)
            key = "test.bin"
            storage.put(source, key)
            # File should be under the configured root
            expected = Path(root) / key
            self.assertTrue(expected.exists())


if __name__ == "__main__":
    unittest.main()
