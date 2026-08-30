from __future__ import annotations

import base64
import json
import tempfile
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_mgmt_service.adapters.mcp.server import MCPServer
from document_mgmt_service.application.access import DocumentAccessService
from document_mgmt_service.application.health import HealthService
from document_mgmt_service.application.ingestion import IngestionDependencies, IngestionService
from document_mgmt_service.application.search import SemanticChunker, SemanticSearchService, TextEmbeddingService
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


class FailingOCR(StaticOCR):
    def extract_text(self, file_path: Path) -> str:
        raise RuntimeError("ocr unavailable")


class DocumentMCPInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

        self.repo = build_postgres_repo(self)
        self.storage = LocalFileStorageAdapter(Path(self.tempdir.name))
        self.store = build_qdrant_store(self, dimension=64)
        self.search = SemanticSearchService(
            store=self.store,
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
            mcp_transport="stdio",
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

    def _build_server(self, ocr: StaticOCR) -> MCPServer:
        ingestion = IngestionService(
            dependencies=IngestionDependencies(
                repository=self.repo,
                storage=self.storage,
                ocr=ocr,
                semantic_search=self.search,
            )
        )
        access = DocumentAccessService(
            repository=self.repo,
            storage=self.storage,
            search=self.search,
        )
        return MCPServer(
            config=self.config,
            health_service=HealthService(self.config),
            ingestion_service=ingestion,
            access_service=access,
        )

    def _rpc(self, server: MCPServer, name: str, arguments: dict[str, object] | None = None, request_id: int = 1) -> dict[str, object]:
        response = server._dispatch(  # noqa: SLF001
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            }
        )
        self.assertIsNotNone(response)
        return response

    def _tool_text(self, response: dict[str, object]) -> dict[str, object]:
        self.assertIn("result", response)
        content = response["result"]["content"]  # type: ignore[index]
        text = content[0]["text"]  # type: ignore[index]
        return json.loads(text)

    def test_mcp_exposes_document_only_tools(self) -> None:
        server = self._build_server(StaticOCR("alpha page one\fpage two beta"))
        response = server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})  # noqa: SLF001
        tools = [tool["name"] for tool in response["result"]["tools"]]  # type: ignore[index]
        self.assertCountEqual(
            tools,
            [
                "upload_document",
                "get_document_metadata",
                "get_document_description",
                "list_documents",
                "search_documents",
                "search_document_content",
                "get_evidence",
                "get_page",
                "get_document",
                "request_sensitive_access",
                "get_field_value",
            ],
        )
        joined = " ".join(tools)
        self.assertNotIn("qdrant", joined.lower())
        self.assertNotIn("postgres", joined.lower())
        self.assertNotIn("ocr", joined.lower())
        self.assertNotIn("storage", joined.lower())

    def test_ingestion_retrieval_and_provenance(self) -> None:
        server = self._build_server(StaticOCR("alpha page one\fpage two beta"))
        uploaded = self._rpc(
            server,
            "upload_document",
            {
                "filename": "sample.pdf",
                "content_base64": base64.b64encode(b"%PDF-1.4 sample").decode("ascii"),
                "content_type": "application/pdf",
                "document_id": "doc-1",
                "privacy": "OPEN",
                "metadata": {"category": "example"},
            },
        )
        uploaded_payload = self._tool_text(uploaded)
        self.assertEqual(uploaded_payload["status"], "successful")
        self.assertNotIn("description", uploaded_payload)
        self.assertNotIn("metadata", uploaded_payload)
        self.assertEqual(uploaded_payload["document_id"], "doc-1")
        self.assertEqual(uploaded_payload["semantic_index_status"], "INDEXED")
        self.assertGreaterEqual(uploaded_payload["chunk_count"], 1)

        metadata = self._tool_text(self._rpc(server, "get_document_metadata", {"document_id": "doc-1", "version": 1}))
        self.assertEqual(metadata["document_id"], "doc-1")
        self.assertEqual(metadata["metadata"]["category"], "example")

        description = self._tool_text(self._rpc(server, "get_document_description", {"document_id": "doc-1", "version": 1}))
        self.assertIn("pdf document", description["description"])

        documents = self._tool_text(self._rpc(server, "list_documents"))
        self.assertEqual(len(documents["results"]), 1)
        self.assertEqual(documents["results"][0]["latest_version"], 1)

        search = self._tool_text(self._rpc(server, "search_documents", {"query": "alpha"}))
        self.assertEqual(search["query"], "alpha")
        self.assertEqual(search["results"][0]["document_id"], "doc-1")
        self.assertIn("provenance", search["results"][0])

        content = self._tool_text(self._rpc(server, "search_document_content", {"query": "beta"}))
        self.assertEqual(content["results"][0]["document_id"], "doc-1")
        self.assertIn("page_number", content["results"][0]["provenance"])

        evidence = self._tool_text(self._rpc(server, "get_evidence", {"document_id": "doc-1", "version": 1, "query": "beta"}))
        self.assertEqual(evidence["results"][0]["document_id"], "doc-1")
        self.assertIn("provenance", evidence["results"][0])

        page = self._tool_text(self._rpc(server, "get_page", {"document_id": "doc-1", "version": 1, "page_number": 2}))
        self.assertEqual(page["page_number"], 2)
        self.assertIn("beta", page["text"])

        document = self._tool_text(self._rpc(server, "get_document", {"document_id": "doc-1", "version": 1}))
        self.assertEqual(base64.b64decode(document["content_base64"]), b"%PDF-1.4 sample")
        self.assertEqual(document["storage_key"], "documents/doc-1/v1/sample.pdf")

    def test_sensitive_access_requires_approval_and_fails_closed(self) -> None:
        server = self._build_server(StaticOCR("employee ssn 123-45-6789 and salary"))
        self._tool_text(
            self._rpc(
                server,
                "upload_document",
                {
                    "filename": "sensitive.pdf",
                    "content_base64": base64.b64encode(b"%PDF sensitive").decode("ascii"),
                    "content_type": "application/pdf",
                    "document_id": "doc-sensitive",
                },
            )
        )

        approval = self._tool_text(
            self._rpc(
                server,
                "request_sensitive_access",
                {
                    "document_id": "doc-sensitive",
                    "version": 1,
                    "intent": "WHOLE_DOCUMENT",
                },
            )
        )
        self.assertEqual(approval["access_action"], "REQUIRE_APPROVAL")
        self.assertTrue(approval["approval_required"])

        denied_document = self._tool_text(
            self._rpc(server, "get_document", {"document_id": "doc-sensitive", "version": 1})
        )
        self.assertEqual(denied_document["access_action"], "REQUIRE_APPROVAL")
        self.assertIsNone(denied_document["content_base64"])

        denied_page = self._tool_text(
            self._rpc(server, "get_page", {"document_id": "doc-sensitive", "version": 1, "page_number": 1})
        )
        self.assertEqual(denied_page["access_action"], "REQUIRE_APPROVAL")
        self.assertNotIn("ssn", json.dumps(denied_page).lower())

        denied_content = self._tool_text(
            self._rpc(server, "search_document_content", {"query": "ssn", "document_id": "doc-sensitive", "version": 1})
        )
        # Content is redacted (text empty) but chunk metadata is returned
        self.assertEqual(len(denied_content["results"]), 1)
        self.assertEqual(denied_content["results"][0]["access_action"], "REQUIRE_APPROVAL")
        self.assertEqual(denied_content["results"][0]["text"], "")

    def test_whole_document_access_returns_original_storage_content(self) -> None:
        server = self._build_server(StaticOCR("alpha page one\fpage two beta"))
        self._tool_text(
            self._rpc(
                server,
                "upload_document",
                {
                    "filename": "original.pdf",
                    "content_base64": base64.b64encode(b"%PDF-orig").decode("ascii"),
                    "content_type": "application/pdf",
                    "document_id": "doc-original",
                    "privacy": "OPEN",
                },
            )
        )

        document = self._tool_text(self._rpc(server, "get_document", {"document_id": "doc-original", "version": 1}))
        self.assertEqual(base64.b64decode(document["content_base64"]), b"%PDF-orig")
        self.assertEqual(document["filename"], "original.pdf")

    def test_dependency_failure_is_closed(self) -> None:
        server = self._build_server(FailingOCR("ignored"))
        response = self._rpc(
            server,
            "upload_document",
            {
                "filename": "broken.pdf",
                "content_base64": base64.b64encode(b"%PDF-broken").decode("ascii"),
                "content_type": "application/pdf",
                "document_id": "doc-broken",
            },
            request_id=77,
        )
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32603)


if __name__ == "__main__":
    unittest.main()
