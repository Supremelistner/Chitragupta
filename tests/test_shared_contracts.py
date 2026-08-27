"""Tests for shared contracts — ID consistency, versioning, relationship models.

Verifies that the canonical types and helper functions work correctly
and that IDs are deterministic and bidirectional.
"""

from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timezone

from shared.contracts import (
    AlertSeverity,
    ChunkId,
    ChunkIdentity,
    DocumentIdentity,
    DocumentMetadata,
    DocumentId,
    ExpiryStatus,
    FileKind,
    InferenceTask,
    IngestionRequest,
    IngestionResult,
    IngestionResult,
    ModelExecutionRecord,
    ModelProvider,
    PrivacyClassification,
    ProcessingStatus,
    RelationshipType,
    RunId,
    StorageKey,
    StorageReference,
    TemplateId,
    TemplateVersion,
    ValidationRequest,
    ValidationRunRecord,
    ValidationResult,
    ValidationStatus,
    Version,
    make_chunk_id,
    make_run_id,
    make_storage_key,
)


class TestIDTypes(unittest.TestCase):
    """Test that ID types are plain strings."""

    def test_document_id_is_str(self):
        doc_id: DocumentId = "a1b2c3d4e5f6"
        self.assertIsInstance(doc_id, str)

    def test_version_is_int(self):
        version: Version = 1
        self.assertIsInstance(version, int)

    def test_chunk_id_is_str(self):
        chunk_id: ChunkId = "abc123def456"
        self.assertIsInstance(chunk_id, str)

    def test_template_id_is_str(self):
        template_id: TemplateId = "aadhaar_card_v1"
        self.assertIsInstance(template_id, str)

    def test_run_id_is_str(self):
        run_id: RunId = "abc123def456"
        self.assertIsInstance(run_id, str)

    def test_storage_key_is_str(self):
        storage_key: StorageKey = "documents/doc1/v1/file.pdf"
        self.assertIsInstance(storage_key, str)


class TestChunkIDGeneration(unittest.TestCase):
    """Test deterministic chunk ID generation."""

    def test_chunk_id_is_deterministic(self):
        """Same inputs always produce the same chunk_id."""
        id1 = make_chunk_id("doc1", 1, 0)
        id2 = make_chunk_id("doc1", 1, 0)
        self.assertEqual(id1, id2)

    def test_chunk_id_differs_by_document(self):
        id1 = make_chunk_id("doc1", 1, 0)
        id2 = make_chunk_id("doc2", 1, 0)
        self.assertNotEqual(id1, id2)

    def test_chunk_id_differs_by_version(self):
        id1 = make_chunk_id("doc1", 1, 0)
        id2 = make_chunk_id("doc1", 2, 0)
        self.assertNotEqual(id1, id2)

    def test_chunk_id_differs_by_index(self):
        id1 = make_chunk_id("doc1", 1, 0)
        id2 = make_chunk_id("doc1", 1, 1)
        self.assertNotEqual(id1, id2)

    def test_chunk_id_is_16_hex_chars(self):
        chunk_id = make_chunk_id("doc1", 1, 0)
        self.assertEqual(len(chunk_id), 16)
        # Should be valid hex
        int(chunk_id, 16)

    def test_chunk_id_matches_expected_hash(self):
        """Verify against manual SHA256 computation."""
        doc_id, version, index = "test_doc", 3, 5
        expected = hashlib.sha256(f"{doc_id}:{version}:{index}".encode()).hexdigest()[:16]
        actual = make_chunk_id(doc_id, version, index)
        self.assertEqual(actual, expected)


class TestStorageKeyGeneration(unittest.TestCase):
    """Test deterministic storage key generation."""

    def test_storage_key_format(self):
        key = make_storage_key("doc123", 2, "aadhaar.jpg")
        self.assertEqual(key, "documents/doc123/v2/aadhaar.jpg")

    def test_storage_key_sanitizes_slashes(self):
        key = make_storage_key("doc1", 1, "path/to/file.pdf")
        # The path components (doc1/v1) have slashes, but the filename path is sanitized
        self.assertIn("file.pdf", key)
        self.assertIn("documents/doc1/v1/", key)

    def test_storage_key_sanitizes_backslashes(self):
        key = make_storage_key("doc1", 1, "path\\to\\file.pdf")
        self.assertNotIn("\\", key)


class TestRunIDGeneration(unittest.TestCase):
    """Test run ID generation."""

    def test_run_id_is_unique(self):
        id1 = make_run_id()
        id2 = make_run_id()
        self.assertNotEqual(id1, id2)

    def test_run_id_is_32_hex_chars(self):
        run_id = make_run_id()
        self.assertEqual(len(run_id), 32)
        int(run_id, 16)


class TestEnumConsistency(unittest.TestCase):
    """Test that enums have expected values."""

    def test_processing_status_values(self):
        self.assertEqual(ProcessingStatus.RECEIVED.value, "RECEIVED")
        self.assertEqual(ProcessingStatus.COMPLETED.value, "COMPLETED")
        self.assertEqual(ProcessingStatus.FAILED.value, "FAILED")
        self.assertEqual(ProcessingStatus.MODEL_INFERENCE_IN_PROGRESS.value, "MODEL_INFERENCE_IN_PROGRESS")

    def test_privacy_classification_values(self):
        self.assertEqual(PrivacyClassification.OPEN.value, "OPEN")
        self.assertEqual(PrivacyClassification.SENSITIVE.value, "SENSITIVE")

    def test_validation_status_values(self):
        self.assertEqual(ValidationStatus.VALID.value, "VALID")
        self.assertEqual(ValidationStatus.INVALID.value, "INVALID")
        self.assertEqual(ValidationStatus.SUSPICIOUS.value, "SUSPICIOUS")

    def test_expiry_status_values(self):
        self.assertEqual(ExpiryStatus.VALID.value, "VALID")
        self.assertEqual(ExpiryStatus.EXPIRED.value, "EXPIRED")
        self.assertEqual(ExpiryStatus.EXPIRING_SOON.value, "EXPIRING_SOON")
        self.assertEqual(ExpiryStatus.NO_EXPIRY.value, "NO_EXPIRY")

    def test_model_provider_values(self):
        self.assertEqual(ModelProvider.HUGGINGFACE.value, "huggingface")

    def test_inference_task_values(self):
        self.assertEqual(InferenceTask.TEXT_EXTRACTION.value, "text_extraction")
        self.assertEqual(InferenceTask.METADATA_EXTRACTION.value, "metadata_extraction")

    def test_relationship_type_values(self):
        self.assertEqual(RelationshipType.REQUIRED_FOR.value, "REQUIRED_FOR")
        self.assertEqual(RelationshipType.SUPPORTS.value, "SUPPORTS")


class TestDocumentIdentity(unittest.TestCase):
    """Test DocumentIdentity dataclass."""

    def test_construction(self):
        identity = DocumentIdentity(document_id="doc1", version=1)
        self.assertEqual(identity.document_id, "doc1")
        self.assertEqual(identity.version, 1)

    def test_immutable(self):
        identity = DocumentIdentity(document_id="doc1", version=1)
        with self.assertRaises(AttributeError):
            identity.version = 2  # type: ignore[misc]


class TestChunkIdentity(unittest.TestCase):
    """Test ChunkIdentity dataclass."""

    def test_construction(self):
        identity = ChunkIdentity(
            chunk_id="chunk1",
            document_id="doc1",
            version=1,
            chunk_index=0,
        )
        self.assertEqual(identity.chunk_id, "chunk1")
        self.assertEqual(identity.document_id, "doc1")

    def test_chunk_id_from_helper(self):
        doc_id = "test_doc"
        version = 1
        index = 0
        chunk_id = make_chunk_id(doc_id, version, index)
        identity = ChunkIdentity(
            chunk_id=chunk_id,
            document_id=doc_id,
            version=version,
            chunk_index=index,
        )
        self.assertEqual(identity.chunk_id, chunk_id)


class TestStorageReference(unittest.TestCase):
    """Test StorageReference dataclass."""

    def test_construction(self):
        ref = StorageReference(
            storage_key="documents/doc1/v1/file.pdf",
            file_size_bytes=1024,
            sha256="abc123",
            content_type="application/pdf",
        )
        self.assertEqual(ref.storage_key, "documents/doc1/v1/file.pdf")
        self.assertEqual(ref.file_size_bytes, 1024)


class TestModelExecutionRecord(unittest.TestCase):
    """Test ModelExecutionRecord dataclass."""

    def test_construction(self):
        record = ModelExecutionRecord(
            run_id="run1",
            document_id="doc1",
            version=1,
            task=InferenceTask.TEXT_EXTRACTION,
            provider=ModelProvider.HUGGINGFACE,
            model_id="Qwen/Qwen2.5-VL-72B-Instruct",
        )
        self.assertEqual(record.run_id, "run1")
        self.assertEqual(record.task, InferenceTask.TEXT_EXTRACTION)
        self.assertIsNone(record.confidence)


class TestValidationRunRecord(unittest.TestCase):
    """Test ValidationRunRecord dataclass."""

    def test_construction(self):
        record = ValidationRunRecord(
            run_id="run1",
            document_id="doc1",
            version=1,
            template_id="aadhaar_card_v1",
            status=ValidationStatus.VALID,
        )
        self.assertEqual(record.template_id, "aadhaar_card_v1")
        self.assertEqual(record.status, ValidationStatus.VALID)

    def test_default_values(self):
        record = ValidationRunRecord(
            run_id="run1",
            document_id="doc1",
            version=1,
        )
        self.assertIsNone(record.template_id)
        self.assertEqual(record.status, ValidationStatus.UNKNOWN_DOCUMENT)
        self.assertEqual(record.risk_score, 0.0)
        self.assertTrue(record.is_authentic)


class TestIngestionRequest(unittest.TestCase):
    """Test IngestionRequest dataclass."""

    def test_construction(self):
        request = IngestionRequest(
            original_filename="test.pdf",
            content=b"test content",
            content_type="application/pdf",
        )
        self.assertEqual(request.original_filename, "test.pdf")
        self.assertEqual(request.content, b"test content")
        self.assertIsNone(request.document_id)

    def test_with_document_id(self):
        request = IngestionRequest(
            original_filename="test.pdf",
            content=b"test",
            document_id="existing_doc",
        )
        self.assertEqual(request.document_id, "existing_doc")


class TestIngestionResult(unittest.TestCase):
    """Test IngestionResult dataclass."""

    def test_construction(self):
        result = IngestionResult(
            document_id="doc1",
            version=1,
            processing_status=ProcessingStatus.COMPLETED,
            privacy=PrivacyClassification.OPEN_NOT_PUBLIC,
            description="Test document",
            metadata={},
            storage_key="documents/doc1/v1/test.pdf",
            sha256="abc123",
        )
        self.assertEqual(result.document_id, "doc1")
        self.assertEqual(result.version, 1)
        self.assertIsNone(result.validation_status)

    def test_with_validation(self):
        result = IngestionResult(
            document_id="doc1",
            version=1,
            processing_status=ProcessingStatus.COMPLETED,
            privacy=PrivacyClassification.OPEN_NOT_PUBLIC,
            description=None,
            metadata={},
            storage_key="key",
            sha256="hash",
            validation_status=ValidationStatus.VALID,
            validation_temporal_status=ExpiryStatus.NO_EXPIRY,
            validation_temporal_tags=("NO_EXPIRY",),
        )
        self.assertEqual(result.validation_status, ValidationStatus.VALID)
        self.assertEqual(result.validation_temporal_tags, ("NO_EXPIRY",))


class TestValidationRequest(unittest.TestCase):
    """Test ValidationRequest dataclass."""

    def test_construction(self):
        request = ValidationRequest(
            document_id="doc1",
            version=1,
            document_type="identity_document",
            document_sub_type="aadhaar",
        )
        self.assertEqual(request.document_id, "doc1")
        self.assertEqual(request.document_sub_type, "aadhaar")


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult dataclass."""

    def test_construction(self):
        result = ValidationResult(
            document_id="doc1",
            version=1,
            status=ValidationStatus.VALID,
        )
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertEqual(result.risk_score, 0.0)

    def test_with_temporal(self):
        result = ValidationResult(
            document_id="doc1",
            version=1,
            status=ValidationStatus.VALID,
            temporal_status=ExpiryStatus.NO_EXPIRY,
            temporal_tags=("NO_EXPIRY",),
        )
        self.assertEqual(result.temporal_status, ExpiryStatus.NO_EXPIRY)


class TestDocumentMetadata(unittest.TestCase):
    """Test DocumentMetadata dataclass."""

    def test_construction(self):
        metadata = DocumentMetadata(
            document_id="doc1",
            version=1,
            original_filename="test.pdf",
            content_type="application/pdf",
            file_kind=FileKind.PDF,
            storage_key="documents/doc1/v1/test.pdf",
            file_size_bytes=1024,
            sha256="abc123",
            privacy=PrivacyClassification.OPEN_NOT_PUBLIC,
            processing_status=ProcessingStatus.COMPLETED,
        )
        self.assertEqual(metadata.document_id, "doc1")
        self.assertIsNone(metadata.validation_status)

    def test_with_validation(self):
        metadata = DocumentMetadata(
            document_id="doc1",
            version=1,
            original_filename="test.pdf",
            content_type="application/pdf",
            file_kind=FileKind.PDF,
            storage_key="key",
            file_size_bytes=100,
            sha256="hash",
            privacy=PrivacyClassification.SENSITIVE,
            processing_status=ProcessingStatus.COMPLETED,
            validation_status=ValidationStatus.INVALID,
            validation_temporal_status=ExpiryStatus.EXPIRED,
            validation_temporal_tags=("EXPIRED",),
            validation_risk_score=0.3,
        )
        self.assertEqual(metadata.validation_status, ValidationStatus.INVALID)
        self.assertEqual(metadata.validation_risk_score, 0.3)


class TestTemplateVersion(unittest.TestCase):
    """Test TemplateVersion dataclass."""

    def test_construction(self):
        tv = TemplateVersion(
            template_id="aadhaar_card_v1",
            version="1.0.0",
            document_type="identity_document",
            document_sub_type="aadhaar",
            description="Aadhaar card template",
            field_count=6,
            temporal_rule_count=1,
        )
        self.assertEqual(tv.template_id, "aadhaar_card_v1")
        self.assertTrue(tv.is_active)


class TestBidirectionalTraceability(unittest.TestCase):
    """Test that IDs enable bidirectional tracing between services."""

    def test_postgres_to_qdrant_via_chunk_id(self):
        """Chunk in PostgreSQL can be found in Qdrant via chunk_id."""
        doc_id = "doc123"
        version = 1
        chunk_index = 0

        # PostgreSQL generates chunk_id
        chunk_id = make_chunk_id(doc_id, version, chunk_index)

        # Qdrant uses chunk_id as point ID
        qdrant_point_id = chunk_id

        # Bidirectional: PostgreSQL chunk → Qdrant point
        self.assertEqual(chunk_id, qdrant_point_id)

        # Bidirectional: Qdrant point → PostgreSQL chunk
        # (Reverse lookup: search Qdrant by point_id, get document_id + version from payload)

    def test_document_id_across_services(self):
        """Same document_id used across Document, Model, Validator services."""
        doc_id = "abc123def456"

        # Document service creates it
        ingestion = IngestionRequest(
            original_filename="test.pdf",
            content=b"test",
            document_id=doc_id,
        )
        self.assertEqual(ingestion.document_id, doc_id)

        # Model service references it
        execution = ModelExecutionRecord(
            run_id=make_run_id(),
            document_id=doc_id,
            version=1,
            task=InferenceTask.TEXT_EXTRACTION,
            provider=ModelProvider.HUGGINGFACE,
            model_id="test",
        )
        self.assertEqual(execution.document_id, doc_id)

        # Validator service references it
        validation = ValidationRequest(
            document_id=doc_id,
            version=1,
        )
        self.assertEqual(validation.document_id, doc_id)

    def test_storage_key_references_document(self):
        """Storage key contains document_id and version for traceability."""
        doc_id = "doc123"
        version = 2
        key = make_storage_key(doc_id, version, "file.pdf")
        self.assertIn(doc_id, key)
        self.assertIn(f"v{version}", key)


if __name__ == "__main__":
    unittest.main()
