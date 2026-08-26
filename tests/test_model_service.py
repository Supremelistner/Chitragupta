"""Tests for Model Service and Document Service OCR/image endpoint additions.

Covers:
- Model service domain models
- HuggingFace provider adapter (unit, no network)
- InferenceService orchestration
- MCP tool listing and dispatch
- Document service OCR endpoint
- Document service image endpoint
- MCP tools for OCR and image retrieval
"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from model_service.domain.models import (
    InferenceRequest,
    InferenceResult,
    InferenceTaskType,
    ModelProviderType,
    ProviderHealth,
)
from model_service.domain.ports import ModelProvider
from model_service.application.inference import InferenceService
from model_service.infrastructure.huggingface import HuggingFaceProviderAdapter
from model_service.infrastructure.memory import InMemoryModelRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockProvider:
    """A mock model provider for testing without network."""

    def __init__(self, output: str = "mock output") -> None:
        self._output = output
        self._call_count = 0

    @property
    def provider_type(self) -> ModelProviderType:
        return ModelProviderType.LOCAL

    def ping(self) -> None:
        return None

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self._call_count += 1
        return InferenceResult(
            task=request.task,
            provider=self.provider_type,
            model_id="mock-model",
            output=self._output,
            request_id=request.request_id,
        )

    def list_models(self):
        return []

    def health(self) -> ProviderHealth:
        return ProviderHealth(provider=self.provider_type, healthy=True)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------

class TestDomainModels(unittest.TestCase):
    def test_inference_request_defaults(self) -> None:
        req = InferenceRequest(task=InferenceTaskType.CUSTOM)
        self.assertEqual(req.task, InferenceTaskType.CUSTOM)
        self.assertIsNone(req.image_bytes)
        self.assertEqual(req.prompt, "")

    def test_inference_result_fields(self) -> None:
        result = InferenceResult(
            task=InferenceTaskType.OCR_VERIFICATION,
            provider=ModelProviderType.HUGGINGFACE,
            model_id="test-model",
            output="verified",
            confidence=0.95,
        )
        self.assertEqual(result.output, "verified")
        self.assertEqual(result.confidence, 0.95)

    def test_task_types_covered(self) -> None:
        tasks = [
            InferenceTaskType.DOCUMENT_CLASSIFICATION,
            InferenceTaskType.OCR_VERIFICATION,
            InferenceTaskType.METADATA_EXTRACTION,
            InferenceTaskType.CONTENT_SUMMARIZATION,
            InferenceTaskType.PRIVACY_CLASSIFICATION,
            InferenceTaskType.CUSTOM,
        ]
        self.assertEqual(len(tasks), 6)


# ---------------------------------------------------------------------------
# InferenceService tests
# ---------------------------------------------------------------------------

class TestInferenceService(unittest.TestCase):
    def setUp(self) -> None:
        self.mock = _MockProvider("test output")
        self.repository = InMemoryModelRepository(initial_provider="mock")
        self.service = InferenceService(
            providers={"mock": self.mock},
            repository=self.repository,
        )

    def test_infer_uses_active_provider(self) -> None:
        result = self.service.infer(InferenceRequest(task=InferenceTaskType.CUSTOM))
        self.assertEqual(result.output, "test output")
        self.assertEqual(result.provider, ModelProviderType.LOCAL)
        self.assertEqual(self.mock._call_count, 1)

    def test_switch_provider(self) -> None:
        mock2 = _MockProvider("provider 2")
        self.service = InferenceService(
            providers={"mock": self.mock, "mock2": mock2},
            repository=self.repository,
        )
        self.service.set_active_provider("mock2")
        result = self.service.infer(InferenceRequest(task=InferenceTaskType.CUSTOM))
        self.assertEqual(result.output, "provider 2")
        self.assertEqual(self.mock._call_count, 0)

    def test_invalid_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.service.set_active_provider("nonexistent")

    def test_list_providers(self) -> None:
        providers = self.service.list_providers()
        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["name"], "mock")
        self.assertTrue(providers[0]["healthy"])

    def test_health_check(self) -> None:
        health = self.service.health()
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["healthy"])


# ---------------------------------------------------------------------------
# HuggingFace adapter tests (unit, no network)
# ---------------------------------------------------------------------------

class TestHuggingFaceAdapter(unittest.TestCase):
    def test_provider_type(self) -> None:
        adapter = HuggingFaceProviderAdapter(model_id="test")
        self.assertEqual(adapter.provider_type, ModelProviderType.HUGGINGFACE)

    def test_list_models(self) -> None:
        adapter = HuggingFaceProviderAdapter(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
        models = adapter.list_models()
        self.assertEqual(len(models), 1)
        self.assertTrue(models[0].supports_vision)
        self.assertIn("vision", models[0].capabilities)

    def test_health_without_token(self) -> None:
        adapter = HuggingFaceProviderAdapter(model_id="test")
        # Should still report healthy (ping checks import, not auth)
        health = adapter.health()
        self.assertTrue(health.healthy)

    def test_system_prompts_not_empty(self) -> None:
        """Prompts should be populated with task-specific instructions."""
        adapter = HuggingFaceProviderAdapter(model_id="test")
        for task in InferenceTaskType:
            prompt = adapter._system_prompt(task)
            self.assertGreater(len(prompt), 50, f"Prompt for {task.value} should be populated")


# ---------------------------------------------------------------------------
# MCP server tool listing tests
# ---------------------------------------------------------------------------

class TestModelMCPServerTools(unittest.TestCase):
    def setUp(self) -> None:
        self.config = type("Config", (), {
            "mcp_server_name": "test-model-service",
            "mcp_server_version": "0.1.0",
        })()
        self.mock = _MockProvider()
        self.repository = InMemoryModelRepository()
        self.inference = InferenceService(
            providers={"mock": self.mock},
            repository=self.repository,
        )

    def _build_server(self):
        from model_service.adapters.mcp.server import MCPServer
        return MCPServer(config=self.config, inference_service=self.inference)

    def test_tools_listed(self) -> None:
        server = self._build_server()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        })
        tool_names = [t["name"] for t in response["result"]["tools"]]
        self.assertIn("classify_document", tool_names)
        self.assertIn("verify_ocr", tool_names)
        self.assertIn("extract_metadata", tool_names)
        self.assertIn("summarize_content", tool_names)
        self.assertIn("list_providers", tool_names)
        self.assertIn("health", tool_names)

    def test_health_tool(self) -> None:
        server = self._build_server()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "health", "arguments": {}},
        })
        content = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(content["status"], "healthy")

    def test_list_providers_tool(self) -> None:
        server = self._build_server()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "list_providers", "arguments": {}},
        })
        content = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(len(content["providers"]), 1)


# ---------------------------------------------------------------------------
# Document service OCR/image endpoint tests
# ---------------------------------------------------------------------------

class TestDocumentOCRAndImageEndpoints(unittest.TestCase):
    """Test the new OCR and image endpoints on the document service MCP server."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

        from document_mgmt_service.infrastructure.memory import InMemoryDocumentRepository
        from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter
        from document_mgmt_service.infrastructure.qdrant import QdrantSemanticChunkStoreAdapter
        from document_mgmt_service.application.search import (
            SemanticChunker, SemanticSearchService, TextEmbeddingService,
        )
        from document_mgmt_service.application.ingestion import IngestionDependencies, IngestionService
        from document_mgmt_service.application.access import DocumentAccessService
        from document_mgmt_service.application.health import HealthService

        self.repo = InMemoryDocumentRepository()
        self.storage = LocalFileStorageAdapter(Path(self.tempdir.name))
        self.store = QdrantSemanticChunkStoreAdapter()
        self.search = SemanticSearchService(
            store=self.store,
            embedder=TextEmbeddingService(dimension=32),
            chunker=SemanticChunker(max_chunk_chars=60, chunk_overlap=10),
        )

        class _OCR:
            def ping(self): return None
            def extract_text(self, fp): return "extracted OCR text from document"
            def close(self): return None

        self.ingestion = IngestionService(
            dependencies=IngestionDependencies(
                repository=self.repo, storage=self.storage,
                ocr=_OCR(), semantic_search=self.search,
            )
        )
        self.access = DocumentAccessService(
            repository=self.repo, storage=self.storage, search=self.search,
        )
        self.config = type("Config", (), {
            "mcp_server_name": "test", "mcp_server_version": "0.1.0",
        })()
        self.health = HealthService(self.config)

    def _build_mcp(self):
        from document_mgmt_service.adapters.mcp.server import MCPServer
        return MCPServer(
            config=self.config,
            health_service=self.health,
            ingestion_service=self.ingestion,
            access_service=self.access,
        )

    def _ingest(self, doc_id: str, filename: str = "test.pdf", content: bytes = b"test"):
        from document_mgmt_service.domain.models import DocumentIngestionRequest
        return self.ingestion.ingest(DocumentIngestionRequest(
            original_filename=filename, content=content,
            content_type="application/pdf", document_id=doc_id,
        ))

    def _rpc(self, server, name: str, args: dict | None = None) -> dict:
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args or {}},
        })
        return json.loads(response["result"]["content"][0]["text"])

    def test_mcp_tools_include_ocr_and_image(self) -> None:
        server = self._build_mcp()
        response = server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tool_names = [t["name"] for t in response["result"]["tools"]]
        self.assertIn("get_document_ocr", tool_names)
        self.assertIn("get_document_image", tool_names)

    def test_get_document_ocr_returns_extracted_text(self) -> None:
        self._ingest("doc-ocr-1")
        server = self._build_mcp()
        result = self._rpc(server, "get_document_ocr", {
            "document_id": "doc-ocr-1", "version": 1,
        })
        self.assertEqual(result["document_id"], "doc-ocr-1")
        self.assertEqual(result["version"], 1)
        self.assertIn("extracted OCR text", result["extracted_text"])
        self.assertEqual(result["processing_status"], "INDEXED")

    def test_get_document_ocr_includes_metadata(self) -> None:
        self._ingest("doc-ocr-2")
        server = self._build_mcp()
        result = self._rpc(server, "get_document_ocr", {
            "document_id": "doc-ocr-2", "version": 1,
        })
        self.assertIn("metadata", result)
        self.assertIn("description", result)
        self.assertIn("privacy", result)
        self.assertIn("file_kind", result)

    def test_get_document_image_returns_content(self) -> None:
        content = b"fake-pdf-content-for-image"
        self._ingest("doc-img-1", content=content)
        server = self._build_mcp()
        result = self._rpc(server, "get_document_image", {
            "document_id": "doc-img-1", "version": 1,
        })
        self.assertEqual(result["document_id"], "doc-img-1")
        self.assertIn("content_base64", result)
        self.assertIsNotNone(result["content_base64"])
        decoded = base64.b64decode(result["content_base64"])
        self.assertEqual(decoded, content)

    def test_get_document_image_includes_access_info(self) -> None:
        self._ingest("doc-img-2")
        server = self._build_mcp()
        result = self._rpc(server, "get_document_image", {
            "document_id": "doc-img-2", "version": 1,
        })
        self.assertIn("access_action", result)
        self.assertIn("access_reason", result)
        self.assertIn("storage_key", result)
        self.assertIn("filename", result)

    def test_nonexistent_document_returns_error(self) -> None:
        server = self._build_mcp()
        response = server._dispatch({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "get_document_ocr", "arguments": {
                "document_id": "nonexistent", "version": 1,
            }},
        })
        self.assertIn("error", response)


if __name__ == "__main__":
    unittest.main()
