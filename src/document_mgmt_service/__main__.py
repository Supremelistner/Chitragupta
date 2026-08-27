from __future__ import annotations

import argparse
import signal
import threading
from typing import Any

from document_mgmt_service.adapters.http.health_server import DocumentManagementHTTPServer
from document_mgmt_service.adapters.mcp.server import MCPServer
from document_mgmt_service.application.access import DocumentAccessService
from document_mgmt_service.application.health import HealthService
from document_mgmt_service.application.ingestion import IngestionDependencies, IngestionService
from document_mgmt_service.application.search import SemanticChunker, SemanticSearchService, TextEmbeddingService
from document_mgmt_service.config import AppConfig
from document_mgmt_service.infrastructure.document_validator import IngestionValidator
from document_mgmt_service.infrastructure.memory import InMemoryDocumentRepository
from document_mgmt_service.infrastructure.model_ocr import ModelOCRAdapter
from document_mgmt_service.infrastructure.ocr import NullOCRAdapter
from document_mgmt_service.infrastructure.postgres import PostgreSQLRepositoryAdapter, create_psycopg_connection_factory
from document_mgmt_service.infrastructure.qdrant import QdrantSemanticChunkStoreAdapter
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter
from document_mgmt_service.logging import configure_logging, install_exception_hooks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document Management Microservice")
    parser.add_argument("--http-host", default=None, help="HTTP server host")
    parser.add_argument("--http-port", type=int, default=None, help="HTTP server port")
    parser.add_argument(
        "--http-only",
        action="store_true",
        help="Run only the HTTP server",
    )
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Run only the MCP server over stdio",
    )
    return parser


def _build_repository(config: AppConfig):
    if config.postgres_dsn:
        repository = PostgreSQLRepositoryAdapter(
            create_psycopg_connection_factory(config.postgres_dsn)
        )
        repository.ensure_schema()
        return repository
    return InMemoryDocumentRepository()


def _build_semantic_search(config: AppConfig) -> SemanticSearchService:
    store = QdrantSemanticChunkStoreAdapter(collection_name=config.qdrant_collection_name)
    return SemanticSearchService(
        store=store,
        embedder=TextEmbeddingService(dimension=config.semantic_embedding_dimension),
        chunker=SemanticChunker(
            max_chunk_chars=config.semantic_chunk_size,
            chunk_overlap=config.semantic_chunk_overlap,
        ),
    )


def _build_model_provider(config: AppConfig):
    """Build model service adapter if HF token is configured."""
    if not config.huggingface_token:
        return None
    try:
        from model_service.infrastructure.huggingface import HuggingFaceProviderAdapter
        from model_service.domain.models import InferenceRequest, InferenceTaskType

        hf_adapter = HuggingFaceProviderAdapter(
            token=config.huggingface_token,
            model_id=config.huggingface_model_id,
            timeout_seconds=config.request_timeout_seconds,
        )

        class _ModelClient:
            """Wraps the HF adapter for text extraction and classification."""
            def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
                request = InferenceRequest(
                    task=InferenceTaskType.TEXT_EXTRACTION,
                    image_bytes=image_bytes,
                    image_mime_type=mime_type,
                )
                result = hf_adapter.infer(request)
                if result.output.startswith("ERROR:"):
                    raise RuntimeError(result.output)
                return result.output

            def classify_document(self, image_bytes: bytes, mime_type: str) -> dict[str, Any]:
                """Classify what the document is using the VL model."""
                import json as _json
                request = InferenceRequest(
                    task=InferenceTaskType.DOCUMENT_CLASSIFICATION,
                    image_bytes=image_bytes,
                    image_mime_type=mime_type,
                )
                result = hf_adapter.infer(request)
                if result.output.startswith("ERROR:"):
                    raise RuntimeError(result.output)
                return _json.loads(result.output)

        return _ModelClient()
    except Exception as exc:
        print(f"Warning: Could not initialize model service: {exc}")
        return None


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig.from_env(
        http_host=args.http_host,
        http_port=args.http_port,
    )
    configure_logging(config.log_level)
    install_exception_hooks()

    repository = _build_repository(config)
    storage = LocalFileStorageAdapter(config.file_storage_root)
    semantic_search = _build_semantic_search(config)

    # Build model-backed OCR or fall back to NullOCR
    model_provider = _build_model_provider(config)
    if model_provider is not None:
        ocr = ModelOCRAdapter(model_provider)
        print(f"Using model service for text extraction: {config.huggingface_model_id}")
    else:
        ocr = NullOCRAdapter()
        print("No model service configured — using NullOCR (no text extraction)")

    # Build document validator
    validator = IngestionValidator()
    print(f"Validator loaded with {len(validator._registry.list_all())} document templates")

    access_service = DocumentAccessService(
        repository=repository,
        storage=storage,
        search=semantic_search,
    )
    ingestion_service = IngestionService(
        dependencies=IngestionDependencies(
            repository=repository,
            storage=storage,
            ocr=ocr,
            semantic_search=semantic_search,
            model_provider=model_provider,
            validator=validator,
        )
    )
    health_service = HealthService(config=config)

    http_server = None
    http_thread = None
    stop_event = threading.Event()

    def shutdown(*_: object) -> None:
        stop_event.set()
        if http_server is not None:
            http_server.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        if not args.mcp_only:
            http_server = DocumentManagementHTTPServer(
                host=config.http_host,
                port=config.http_port,
                health_service=health_service,
                ingestion_service=ingestion_service,
                access_service=access_service,
            )
            http_thread = threading.Thread(
                target=http_server.serve_forever,
                name="document-http-server",
                daemon=True,
            )
            http_thread.start()

        if not args.http_only:
            MCPServer(
                config=config,
                health_service=health_service,
                ingestion_service=ingestion_service,
                access_service=access_service,
            ).serve()
        else:
            stop_event.wait()
    finally:
        if http_server is not None:
            http_server.stop()
        if http_thread is not None:
            http_thread.join(timeout=5)
        repository.close()
        storage.close()
        ocr.close()
        # In-memory semantic store does not require shutdown, but the adapter remains swappable.


if __name__ == "__main__":
    main()
