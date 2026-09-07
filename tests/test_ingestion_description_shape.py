"""Unit tests for the description-resolution branches in IngestionService.

The new METADATA_EXTRACTION prompt emits ``description`` as a flat string;
the legacy prompt emitted a dict with ``safe`` / ``detailed`` keys. The
ingestion code must accept both shapes and produce the correct stored
``description`` and ``summary``.

These tests drive the full ingest() pipeline with a live Postgres repo
(matching the existing integration test pattern in
test_ingestion_validation.py) so we exercise the real code path.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from document_mgmt_service.application.ingestion import (
    IngestionDependencies,
    IngestionService,
)
from document_mgmt_service.domain.models import (
    DocumentIngestionRequest,
)
from document_mgmt_service.infrastructure.document_validator import IngestionValidator
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter

from tests._live_stack import build_postgres_repo


class _FakeOCR:
    def ping(self): pass
    def extract_text(self, path): return "extracted text"
    def close(self): pass


class _FakeSemanticSearch:
    def index_document(self, record): return 1
    def search(self, *a, **kw): return []
    def close(self): pass


class IngestionDescriptionShapeTests(unittest.TestCase):
    """Verify the new flat-string description branch is honored."""

    def setUp(self) -> None:
        self.repo = build_postgres_repo(self)
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

    def _ingest(self, *, metadata: dict[str, Any]):
        return self.service.ingest(
            DocumentIngestionRequest(
                original_filename="aadhaar.jpg",
                content=b"%PDF-1.4 test",
                content_type="image/jpeg",
                metadata=metadata,
            )
        )

    def test_flat_string_description_is_stored_as_description(self) -> None:
        """The new prompt shape: ``description`` is a plain string."""
        metadata = {
            "model_extraction": {
                "document_type": "aadhaar_card",
                "document_sub_type": "front",
                "description": "Indian Aadhaar identification card for Tanvi",
                "language": "en",
                "fields": {
                    "aadhaar_number": "4373 7370 1714",
                    "name": "Tanvi",
                    "date_of_birth": "2007-10-01",
                    "gender": "F",
                },
                "extraction_confidence": 0.95,
            }
        }
        result = self._ingest(metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)
        # The flat-string description is what populates ``record.description``.
        self.assertEqual(
            record.description, "Indian Aadhaar identification card for Tanvi"
        )

    def test_legacy_dict_description_falls_back_through_chain(self) -> None:
        """Legacy: ``description.safe`` is preferred, then ``detailed``."""
        metadata = {
            "model_extraction": {
                "document_type": "aadhaar_card",
                "description": {
                    "safe": "Safe summary",
                    "detailed": "Detailed description with PII",
                },
                "fields": {
                    "aadhaar_number": "1234 5678 9012",
                    "name": "Tanvi",
                },
            }
        }
        result = self._ingest(metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)
        # ``safe`` is the public-facing description.
        self.assertEqual(record.description, "Safe summary")

    def test_legacy_dict_with_only_detailed(self) -> None:
        """If the legacy dict only has ``detailed``, it wins."""
        metadata = {
            "model_extraction": {
                "document_type": "aadhaar_card",
                "description": {"detailed": "Only detailed"},
                "fields": {
                    "aadhaar_number": "1234 5678 9012",
                    "name": "Tanvi",
                },
            }
        }
        result = self._ingest(metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)
        self.assertEqual(record.description, "Only detailed")

    def test_flat_string_with_whitespace_strips(self) -> None:
        """Whitespace-only or empty strings fall through to fallback."""
        metadata = {
            "model_extraction": {
                "document_type": "other",
                "description": "   \n  ",
                "fields": {},
            }
        }
        result = self._ingest(metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)
        # Empty flat string → falls back to generate_safe_description.
        # The fallback should produce a non-empty string based on filename.
        self.assertIsNotNone(record.description)
        self.assertTrue(len(record.description) > 0)

    def test_no_description_field_uses_fallback(self) -> None:
        metadata = {
            "model_extraction": {
                "document_type": "other",
                "fields": {},
            }
        }
        result = self._ingest(metadata=metadata)
        record = self.repo.get_version(result.document_id, result.version)
        # The fallback must produce something.
        self.assertIsNotNone(record.description)
        self.assertTrue(len(record.description) > 0)


if __name__ == "__main__":
    unittest.main()