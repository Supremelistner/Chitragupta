"""Tests for the get_field_value tool (two-step confirmation + PII redaction)."""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from document_mgmt_service.adapters.mcp.http_server import build_app
from document_mgmt_service.adapters.mcp.server import MCPServer
from document_mgmt_service.application.access import DocumentAccessService
from document_mgmt_service.application.fields import (
    FieldAccessError,
    FieldValueResult,
    get_field_value,
)
from document_mgmt_service.application.health import HealthService
from document_mgmt_service.application.ingestion import IngestionDependencies, IngestionService
from document_mgmt_service.application.search import (
    SemanticChunker,
    SemanticSearchService,
    TextEmbeddingService,
)
from document_mgmt_service.config import AppConfig
from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentPrivacyClassification,
    DocumentProcessingStatus,
    DocumentVersionRecord,
    SemanticIndexStatus,
)
from document_mgmt_service.infrastructure.memory import InMemoryDocumentRepository
from document_mgmt_service.infrastructure.qdrant import QdrantSemanticChunkStoreAdapter
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter


def _make_record(
    document_id: str = "doc-1",
    *,
    extracted_fields=None,
    owner_type=None,
    relation=None,
    relation_name=None,
    description_safe=None,
    summary=None,
):
    return DocumentVersionRecord(
        document_id=document_id,
        version=1,
        original_filename="aadhaar.jpg",
        content_type="image/jpeg",
        file_kind=DocumentFileKind.IMAGE,
        storage_key="documents/" + document_id + "/v1/aadhaar.jpg",
        file_size_bytes=1024,
        sha256="abc",
        privacy=DocumentPrivacyClassification.SENSITIVE,
        processing_status=DocumentProcessingStatus.INDEXED,
        semantic_index_status=SemanticIndexStatus.INDEXED,
        owner_type=owner_type,
        relation=relation,
        relation_name=relation_name,
        summary=summary,
        description_safe=description_safe,
        extracted_fields=extracted_fields,
    )


class GetFieldValueUnitTests(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryDocumentRepository()
        self.repo.upsert_version(_make_record(
            extracted_fields={"aadhaar_number": "1234 5678 9012", "name": "Manish"},
            owner_type="SELF",
            description_safe="Aadhaar card",
            summary="Aadhaar card",
        ))

    def test_first_call_requires_confirmation(self):
        result = get_field_value(
            self.repo, document_id="doc-1", version=1, field_name="aadhaar_number",
        )
        self.assertIsInstance(result, FieldValueResult)
        self.assertEqual(result.status, "requires_confirmation")
        self.assertIsNone(result.value)
        self.assertIsNotNone(result.source)
        self.assertEqual(result.source.name, "Aadhaar card")
        self.assertEqual(result.source.relation, "SELF")
        self.assertIn("Confirm to see the value", result.suggestion or "")

    def test_second_call_returns_value(self):
        result = get_field_value(
            self.repo, document_id="doc-1", version=1, field_name="aadhaar_number",
            confirm=True,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.value, "1234 5678 9012")
        self.assertIsNotNone(result.source)
        self.assertIn("Retrieve original file", result.suggestion or "")

    def test_unknown_field_returns_not_found(self):
        result = get_field_value(
            self.repo, document_id="doc-1", version=1, field_name="pan_number",
        )
        self.assertEqual(result.status, "not_found")
        self.assertIn("not extracted", result.suggestion or "")
        self.assertIn("aadhaar_number", result.available_fields or [])

    def test_missing_document_raises(self):
        with self.assertRaises(FieldAccessError):
            get_field_value(
                self.repo, document_id="nonexistent", version=1, field_name="x",
            )

    def test_source_falls_back_to_filename(self):
        # No description_safe / summary set → falls back to original filename
        self.repo.upsert_version(_make_record(
            document_id="doc-3",
            extracted_fields={"x": "1"},
            owner_type="SELF",
        ))
        result = get_field_value(
            self.repo, document_id="doc-3", version=1, field_name="x",
        )
        self.assertEqual(result.source.name, "aadhaar.jpg")

    def test_source_relation_name_propagates(self):
        self.repo.upsert_version(_make_record(
            document_id="doc-2",
            extracted_fields={"aadhaar_number": "9999"},
            owner_type="OTHER",
            relation="Mother",
            relation_name="Priya",
            description_safe="Mother's Aadhaar",
        ))
        result = get_field_value(
            self.repo, document_id="doc-2", version=1, field_name="aadhaar_number",
        )
        self.assertEqual(result.source.relation, "OTHER")
        self.assertEqual(result.source.relation_name, "Priya")
        self.assertEqual(result.source.name, "Mother's Aadhaar")


class GetFieldValueMCPTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        repo = InMemoryDocumentRepository()
        repo.upsert_version(_make_record(
            document_id="mcp-test",
            extracted_fields={"aadhaar_number": "1234 5678 9012", "name": "Manish Rana"},
            owner_type="SELF",
            description_safe="Aadhaar card",
            summary="Aadhaar card",
        ))
        storage = LocalFileStorageAdapter(Path(self.tempdir.name))
        store = QdrantSemanticChunkStoreAdapter()
        search = SemanticSearchService(
            store=store,
            embedder=TextEmbeddingService(dimension=64),
            chunker=SemanticChunker(max_chunk_chars=120, chunk_overlap=20),
        )
        ingestion = IngestionService(
            dependencies=IngestionDependencies(
                repository=repo, storage=storage, semantic_search=search,
            )
        )
        access = DocumentAccessService(repository=repo, storage=storage, search=search)
        config = AppConfig(
            app_name="document-management-service",
            environment="test", log_level="INFO",
            http_host="127.0.0.1", http_port=8080,
            mcp_server_name="document-management-service", mcp_server_version="0.1.0",
            mcp_transport="http", mcp_http_host="127.0.0.1", mcp_http_port=8085,
            postgres_dsn=None, qdrant_url=None, qdrant_collection_name="document_chunks",
            semantic_embedding_dimension=64, semantic_chunk_size=120, semantic_chunk_overlap=20,
            file_storage_root=Path(self.tempdir.name), sqlite_db_path=None,
            encryption_master_key="", ocr_enabled=True,
            huggingface_token=None, huggingface_model_id="test",
            request_timeout_seconds=30,
        )
        mcp = MCPServer(
            config=config, health_service=HealthService(config),
            ingestion_service=ingestion, access_service=access,
        )
        self.client = TestClient(build_app(mcp))

    def _call(self, args):
        resp = self.client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "get_field_value", "arguments": args}},
        )
        return resp.json()

    def test_two_step_protocol(self):
        first = self._call({"document_id": "mcp-test", "version": 1, "field": "aadhaar_number"})
        first_text = json.loads(first["result"]["content"][0]["text"])
        self.assertEqual(first_text["status"], "requires_confirmation")
        self.assertIsNone(first_text["value"])
        self.assertEqual(first_text["source"]["name"], "Aadhaar card")

        second = self._call({
            "document_id": "mcp-test", "version": 1,
            "field": "aadhaar_number", "confirm": True,
        })
        second_text = json.loads(second["result"]["content"][0]["text"])
        self.assertEqual(second_text["status"], "ok")
        self.assertEqual(second_text["value"], "1234 5678 9012")
        self.assertIn("Retrieve original file", second_text["suggestion"])

    def test_response_has_no_url(self):
        first = self._call({"document_id": "mcp-test", "version": 1, "field": "name"})
        first_text = json.loads(first["result"]["content"][0]["text"])
        source = first_text["source"]
        self.assertNotIn("url", source)
        self.assertNotIn("document_url", source)
        self.assertNotIn("download_url", source)

    def test_unknown_field_helpful_suggestion(self):
        result = self._call({"document_id": "mcp-test", "version": 1, "field": "pan_number"})
        text = json.loads(result["result"]["content"][0]["text"])
        self.assertEqual(text["status"], "not_found")
        self.assertIn("aadhaar_number", text["available_fields"])


if __name__ == "__main__":
    unittest.main()
