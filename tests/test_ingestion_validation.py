"""Integration test: validation during ingestion.

Verifies that:
1. Every upload goes through validation
2. Valid documents get validation_status=VALID
3. Invalid/fake documents get flagged with alerts
4. Validation metadata is persisted in the document record
5. Alert payload is stored in metadata for downstream consumption
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from document_mgmt_service.application.ingestion import (
    IngestionDependencies,
    IngestionError,
    IngestionService,
)
from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentIngestionRequest,
    DocumentProcessingStatus,
)
from document_mgmt_service.infrastructure.document_validator import IngestionValidator
from document_mgmt_service.infrastructure.memory import InMemoryDocumentRepository
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter


class _FakeOCR:
    def ping(self): pass
    def extract_text(self, path): return "extracted text content"
    def close(self): pass


class _FakeSemanticSearch:
    def index_document(self, record): return 1
    def search(self, *a, **kw): return []
    def close(self): pass


class TestIngestionValidation(unittest.TestCase):
    """Test that validation runs during ingestion."""

    def setUp(self):
        self.repo = InMemoryDocumentRepository()
        self.storage = LocalFileStorageAdapter(Path(tempfile.mkdtemp()))
        self.ocr = _FakeOCR()
        self.search = _FakeSemanticSearch()
        self.validator = IngestionValidator()

        self.service = IngestionService(
            dependencies=IngestionDependencies(
                repository=self.repo,
                storage=self.storage,
                ocr=self.ocr,
                semantic_search=self.search,
                validator=self.validator,
            ),
        )

    def _ingest(
        self,
        filename: str = "test.pdf",
        content: bytes = b"%PDF-1.4 test content",
        content_type: str = "application/pdf",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self.service.ingest(
            DocumentIngestionRequest(
                original_filename=filename,
                content=content,
                content_type=content_type,
                metadata=metadata or {},
            )
        )

    def test_valid_aadhaar_gets_valid_status(self):
        """A well-formed Aadhaar metadata should pass validation."""
        metadata = {
            "model_extraction": {
                "document_type": "identity_document",
                "document_sub_type": "aadhaar",
                "fields": {
                    "Aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98},
                    "name": {"value": "Tanvi", "confidence": 0.98},
                    "date_of_birth": {"value": "01/10/2007", "confidence": 0.95},
                    "gender": {"value": "FEMALE", "confidence": 0.99},
                },
            }
        }
        result = self._ingest(filename="aadhaar.jpg", metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)

        self.assertEqual(record.metadata.get("validation_status"), "VALID")
        self.assertEqual(record.metadata.get("validation_risk_score"), 0.0)
        self.assertIsNone(record.metadata.get("validation_alert"))

    def test_fake_aadhaar_gets_flagged(self):
        """An incomplete/wrong Aadhaar should be flagged as INVALID."""
        metadata = {
            "model_extraction": {
                "document_type": "identity_document",
                "document_sub_type": "aadhaar",
                "fields": {
                    "name": {"value": "X", "confidence": 0.3},
                    "Aadhaar_number": {"value": "123", "confidence": 0.1},
                },
            }
        }
        result = self._ingest(filename="fake_aadhaar.jpg", metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)

        self.assertEqual(record.metadata.get("validation_status"), "INVALID")
        self.assertGreater(record.metadata.get("validation_risk_score", 0), 0)
        self.assertIsNotNone(record.metadata.get("validation_alert"))
        self.assertGreater(len(record.metadata.get("validation_violations", [])), 0)

    def test_valid_marksheet_gets_valid_status(self):
        """A well-formed marksheet should pass validation."""
        metadata = {
            "model_extraction": {
                "document_type": "academic_record",
                "document_sub_type": "marksheet",
                "fields": {
                    "student_name": {"value": "Tanvi", "confidence": 0.98},
                    "roll_number": {"value": "17266193", "confidence": 0.99},
                    "mother_name": {"value": "Poonam", "confidence": 0.95},
                    "father_name": {"value": "Sanjay Kumar", "confidence": 0.98},
                    "date_of_birth": {"value": "01/10/2007", "confidence": 0.97},
                    "school": {"value": "J N V Paprola", "confidence": 0.99},
                    "result": {"value": "PASS", "confidence": 1.0},
                },
            }
        }
        result = self._ingest(filename="marksheet.pdf", metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)

        self.assertEqual(record.metadata.get("validation_status"), "VALID")
        self.assertEqual(record.metadata.get("validation_matched_template"), "marksheet_v1")

    def test_unknown_type_gets_unknown_document_status(self):
        """A document with no matching template gets UNKNOWN_DOCUMENT."""
        metadata = {
            "model_extraction": {
                "document_type": "medical_record",
                "document_sub_type": "prescription",
                "fields": {},
            }
        }
        result = self._ingest(filename="prescription.pdf", metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)

        self.assertEqual(record.metadata.get("validation_status"), "UNKNOWN_DOCUMENT")
        self.assertIsNotNone(record.metadata.get("validation_alert"))

    def test_no_model_extraction_still_runs_validation(self):
        """Documents without model extraction still get validated (with empty fields)."""
        result = self._ingest(filename="plain.txt", content=b"hello world")
        record = self.repo.get_version(result.document_id, result.version)
        # Validation should have run — even if no type matched
        self.assertIn("validation_status", record.metadata)

    def test_document_still_stored_even_if_invalid(self):
        """Invalid documents are NOT rejected — they're stored and flagged."""
        metadata = {
            "model_extraction": {
                "document_type": "identity_document",
                "document_sub_type": "aadhaar",
                "fields": {},
            }
        }
        result = self._ingest(filename="bad_aadhaar.jpg", metadata=metadata)
        # Document should still be stored
        record = self.repo.get_version(result.document_id, result.version)
        self.assertIsNotNone(record)
        self.assertIn(result.document_id, [d.document_id for d in self.repo.list_documents()])
        # But flagged
        self.assertEqual(record.metadata.get("validation_status"), "INVALID")

    def test_validation_without_validator_still_works(self):
        """Ingestion works fine when no validator is configured (backward compat)."""
        service_no_validator = IngestionService(
            dependencies=IngestionDependencies(
                repository=self.repo,
                storage=self.storage,
                ocr=self.ocr,
                semantic_search=self.search,
                validator=None,  # No validator
            ),
        )
        result = service_no_validator.ingest(
            DocumentIngestionRequest(
                original_filename="test.pdf",
                content=b"%PDF-1.4 test",
                content_type="application/pdf",
            )
        )
        record = self.repo.get_version(result.document_id, result.version)
        # Should NOT have validation fields
        self.assertNotIn("validation_status", record.metadata)

    def test_suspicious_document_gets_warning(self):
        """A document with some violations gets SUSPICIOUS status."""
        metadata = {
            "model_extraction": {
                "document_type": "identity_document",
                "document_sub_type": "aadhaar",
                "fields": {
                    "Aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98},
                    "name": {"value": "Tanvi", "confidence": 0.98},
                    "date_of_birth": {"value": "2007-10-01", "confidence": 0.9},  # Wrong format
                    "gender": {"value": "FEMALE", "confidence": 0.99},
                },
            }
        }
        result = self._ingest(filename="aadhaar.jpg", metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)

        # Should be flagged due to wrong date format
        status = record.metadata.get("validation_status")
        self.assertIn(status, ("SUSPICIOUS", "INVALID"))
        self.assertIsNotNone(record.metadata.get("validation_alert"))

    def test_validation_alert_contains_risk_score(self):
        """Alert payload includes risk score and violation details."""
        metadata = {
            "model_extraction": {
                "document_type": "identity_document",
                "document_sub_type": "aadhaar",
                "fields": {"name": {"value": "X", "confidence": 0.1}},
            }
        }
        result = self._ingest(filename="bad.jpg", metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)

        alert = record.metadata.get("validation_alert", {})
        self.assertIn("risk_score", alert)
        self.assertIn("violations", alert)
        self.assertIn("alert_type", alert)
        self.assertEqual(alert["alert_type"], "structure_mismatch")
        self.assertGreater(alert["risk_score"], 0)

    def test_all_uploads_get_validation_fields(self):
        """Every upload — regardless of content — gets validation metadata."""
        for filename, content in [
            ("doc1.pdf", b"%PDF-1.4 test"),
            ("doc2.jpg", b"\xff\xd8\xff\xe0"),
            ("doc3.txt", b"plain text"),
        ]:
            result = self._ingest(filename=filename, content=content)
            record = self.repo.get_version(result.document_id, result.version)
            self.assertIn("validation_status", record.metadata, f"Missing validation_status for {filename}")
            self.assertIn("validation_risk_score", record.metadata, f"Missing validation_risk_score for {filename}")


if __name__ == "__main__":
    unittest.main()
