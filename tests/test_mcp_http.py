"""Tests for the HTTP transport of the Document Management MCP server."""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from document_mgmt_service.adapters.mcp.http_server import build_app
from document_mgmt_service.adapters.mcp.server import MCPServer
from document_mgmt_service.application.access import DocumentAccessService
from document_mgmt_service.application.health import HealthService
from document_mgmt_service.application.ingestion import (
    IngestionDependencies,
    IngestionService,
)
from document_mgmt_service.application.search import (
    SemanticChunker,
    SemanticSearchService,
    TextEmbeddingService,
)
from document_mgmt_service.config import AppConfig
from document_mgmt_service.infrastructure.qdrant import QdrantSemanticChunkStoreAdapter
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter

from tests._live_stack import build_postgres_repo, build_qdrant_store


class StaticOCR:
    def __init__(self, text: str) -> None:
        self._text = text

    def ping(self) -> None:
        return None

    def extract_text(self, file_path: Path) -> str:
        return self._text

    def close(self) -> None:
        return None


class DocumentHTTPTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

        repo = build_postgres_repo(self)
        storage = LocalFileStorageAdapter(Path(self.tempdir.name))
        store = build_qdrant_store(self, dimension=64)
        search = SemanticSearchService(
            store=store,
            embedder=TextEmbeddingService(dimension=64),
            chunker=SemanticChunker(max_chunk_chars=120, chunk_overlap=20),
        )
        self.config = AppConfig(
            app_name="document-management-service",
            environment="test",
            log_level="INFO",
            http_host="127.0.0.1",
            http_port=8080,
            mcp_server_name="document-management-service",
            mcp_server_version="0.1.0",
            mcp_transport="http",
            mcp_http_host="127.0.0.1",
            mcp_http_port=8085,
            postgres_dsn=None,
            qdrant_url=None,
            qdrant_collection_name="document_chunks",
            semantic_embedding_dimension=64,
            semantic_chunk_size=120,
            semantic_chunk_overlap=20,
            file_storage_root=Path(self.tempdir.name),
            encryption_master_key="",
            ocr_enabled=True,
            huggingface_token=None,
            huggingface_model_id="test-model",
            request_timeout_seconds=30,
        )
        ingestion = IngestionService(
            dependencies=IngestionDependencies(
                repository=repo,
                storage=storage,
                ocr=StaticOCR("hello world"),
                semantic_search=search,
            )
        )
        access = DocumentAccessService(repository=repo, storage=storage, search=search)
        mcp_server = MCPServer(
            config=self.config,
            health_service=HealthService(self.config),
            ingestion_service=ingestion,
            access_service=access,
        )
        self.client = TestClient(build_app(mcp_server))

    def test_health_endpoint(self) -> None:
        response = self.client.get("/mcp/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["transport"], "http")
        self.assertEqual(body["server"]["name"], "document-management-service")

    def test_root_endpoint_advertises_mcp(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mcp"]["endpoint"], "/mcp")
        self.assertEqual(body["mcp"]["transport"], "http")

    def test_initialize(self) -> None:
        response = self.client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["id"], 1)
        self.assertIn("result", body)
        self.assertEqual(body["result"]["protocolVersion"], "2024-11-05")

    def test_tools_list(self) -> None:
        response = self.client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        tool_names = {tool["name"] for tool in body["result"]["tools"]}
        self.assertIn("upload_document", tool_names)
        self.assertIn("list_documents", tool_names)
        self.assertIn("search_documents", tool_names)
        self.assertNotIn("raw_qdrant_query", tool_names)  # MCP tool boundary

    def test_upload_then_list(self) -> None:
        upload_payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "upload_document",
                "arguments": {
                    "filename": "http-test.pdf",
                    "content_base64": base64.b64encode(b"%PDF-test").decode("ascii"),
                    "content_type": "application/pdf",
                    "document_id": "doc-http-001",
                    "privacy": "OPEN",
                },
            },
        }
        response = self.client.post("/mcp", json=upload_payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("error", body)
        text = body["result"]["content"][0]["text"]
        upload_result = json.loads(text)
        self.assertEqual(upload_result["status"], "successful")
        self.assertEqual(upload_result["document_id"], "doc-http-001")

        # Now list documents and verify the upload is visible.
        list_response = self.client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "list_documents", "arguments": {}},
            },
        )
        self.assertEqual(list_response.status_code, 200)
        list_body = list_response.json()
        list_text = list_body["result"]["content"][0]["text"]
        listed = json.loads(list_text)
        ids = {doc["document_id"] for doc in listed["results"]}
        self.assertIn("doc-http-001", ids)

    def test_invalid_json_returns_parse_error(self) -> None:
        response = self.client.post(
            "/mcp",
            content="not json at all",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["code"], -32700)

    def test_notification_returns_202(self) -> None:
        # A JSON-RPC message without an "id" is a notification.
        # The server should return 202 Accepted with no body.
        response = self.client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "ping"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.content, b"")

    def test_batch_request(self) -> None:
        response = self.client.post(
            "/mcp",
            json=[
                {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "method": "ping"},  # notification
            ],
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, list)
        self.assertEqual(len(body), 2)  # notification omitted
        ids = {item["id"] for item in body}
        self.assertEqual(ids, {1, 2})


if __name__ == "__main__":
    unittest.main()
