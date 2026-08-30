from __future__ import annotations

import argparse
import os
import signal
import threading
from typing import Any

from document_mgmt_service.adapters.http.health_server import DocumentManagementHTTPServer
from document_mgmt_service.adapters.mcp.server import MCPServer
from document_mgmt_service.application.access import DocumentAccessService
from document_mgmt_service.application.health import DependencyState, HealthService
from document_mgmt_service.application.ingestion import IngestionDependencies, IngestionService
from document_mgmt_service.application.search import SemanticChunker, SemanticSearchService, TextEmbeddingService
from document_mgmt_service.config import AppConfig
from document_mgmt_service.infrastructure.document_validator import IngestionValidator
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
        help="Run only the REST HTTP server",
    )
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Run only the MCP server (use --mcp-transport to pick stdio or http)",
    )
    parser.add_argument(
        "--mcp-transport",
        choices=("stdio", "http"),
        default=None,
        help="MCP transport (default: from DOCUMENT_SERVICE_MCP_TRANSPORT, else 'stdio')",
    )
    parser.add_argument(
        "--mcp-http-host",
        default=None,
        help="MCP HTTP transport host (default: DOCUMENT_SERVICE_MCP_HTTP_HOST)",
    )
    parser.add_argument(
        "--mcp-http-port",
        type=int,
        default=None,
        help="MCP HTTP transport port (default: DOCUMENT_SERVICE_MCP_HTTP_PORT)",
    )
    return parser


def _ping_dependency(name: str, target):
    """Call ``target.ping()`` and convert the result into a DependencyState."""
    ping = getattr(target, "ping", None)
    if ping is None:
        return DependencyState(name=name, healthy=True, details={"note": "no ping()"})
    try:
        ping()
        return DependencyState(name=name, healthy=True)
    except Exception as exc:
        return DependencyState(
            name=name,
            healthy=False,
            details={"error": f"{exc.__class__.__name__}: {exc}"},
        )


def _build_repository(config: AppConfig):
    """Build the document repository. Postgres via Docker is the only
    supported backend. If the connection fails, fail fast — the operator
    should bring the Postgres container up rather than silently degrade.
    """
    if not config.postgres_dsn:
        raise SystemExit(
            "DOCUMENT_SERVICE_POSTGRES_DSN is not set. "
            "The document service requires a Postgres connection (typically "
            "the docker compose stack on localhost:5432). "
            "Run `python activate.py --bg` to start the infrastructure, "
            "or set the env var to your own Postgres URL."
        )
    repository = PostgreSQLRepositoryAdapter(
        create_psycopg_connection_factory(config.postgres_dsn)
    )
    repository.ensure_schema()
    print(f"Using PostgreSQL: {config.postgres_dsn}")
    return repository, _ping_dependency("postgres", repository)


def _build_semantic_search(config: AppConfig) -> tuple[SemanticSearchService, DependencyState]:
    """Build the semantic search service. Qdrant via Docker is the only
    supported vector store. The adapter is always in strict mode — a
    broken Qdrant container must surface as an error, never silently
    fall back to a local in-memory store.
    """
    if not config.qdrant_url:
        raise SystemExit(
            "DOCUMENT_SERVICE_QDRANT_URL is not set. "
            "The document service requires a Qdrant connection (typically "
            "the docker compose stack on http://localhost:6333). "
            "Run `python activate.py --bg` to start the infrastructure, "
            "or set the env var to your own Qdrant URL."
        )
    store = QdrantSemanticChunkStoreAdapter(
        base_url=config.qdrant_url,
        collection_name=config.qdrant_collection_name,
        encryption_key=config.encryption_master_key,
        dimension=config.semantic_embedding_dimension,
        strict=True,
    )
    enc_status = "encrypted" if config.encryption_master_key else "plaintext"
    print(f"Using Qdrant: {config.qdrant_url} (payloads {enc_status}, strict)")
    state = _ping_dependency("qdrant", store)
    return SemanticSearchService(
        store=store,
        embedder=TextEmbeddingService(dimension=config.semantic_embedding_dimension),
        chunker=SemanticChunker(
            max_chunk_chars=config.semantic_chunk_size,
            chunk_overlap=config.semantic_chunk_overlap,
        ),
    ), state


def _build_model_provider(config: AppConfig):
    """Build model service adapter if HF token is configured.

    When a Groq API key is also available, wraps the primary HuggingFace
    adapter in a :class:`FallbackProvider` so that 402 (Payment Required),
    429 (Rate Limit), timeouts, and 5xx errors automatically fall through
    to Groq. This keeps the ingestion pipeline operational when HF
    inference credits are exhausted.
    """
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

        # Wrap with Groq fallback when both are available
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                from model_service.infrastructure.groq_provider import GroqProviderAdapter
                from model_service.infrastructure.fallback_provider import FallbackProvider
                groq = GroqProviderAdapter(
                    api_key=groq_api_key,
                    model_id=os.environ.get("GROQ_VL_MODEL", "qwen/qwen3.8-27b"),
                    timeout_seconds=config.request_timeout_seconds,
                )
                hf_adapter = FallbackProvider(primary=hf_adapter, fallback=groq)
                print("  Model fallback: Groq (activates on HF 402/429/timeout/5xx)")
            except Exception as exc:
                print(f"  Warning: could not initialize Groq fallback: {exc}")

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

            def extract_metadata(self, image_bytes: bytes, mime_type: str) -> dict[str, Any]:
                """Pull structured metadata (fields, ownership, expiry) from the document.

                Uses the METADATA_EXTRACTION task, which now also returns the
                owner / relation / expiry block.
                """
                import json as _json
                request = InferenceRequest(
                    task=InferenceTaskType.METADATA_EXTRACTION,
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
    # CLI overrides for MCP transport
    if args.mcp_transport is not None:
        object.__setattr__(config, "mcp_transport", args.mcp_transport)
    if args.mcp_http_host is not None:
        object.__setattr__(config, "mcp_http_host", args.mcp_http_host)
    if args.mcp_http_port is not None:
        object.__setattr__(config, "mcp_http_port", args.mcp_http_port)
    configure_logging(config.log_level)
    install_exception_hooks()

    repository, repo_state = _build_repository(config)
    storage = LocalFileStorageAdapter(config.file_storage_root)
    semantic_search, qdrant_state = _build_semantic_search(config)

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
    health_service = HealthService(config=config, dependencies=[repo_state, qdrant_state])

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
            mcp_server = MCPServer(
                config=config,
                health_service=health_service,
                ingestion_service=ingestion_service,
                access_service=access_service,
            )
            if config.is_mcp_http():
                from document_mgmt_service.adapters.mcp.http_server import run as run_mcp_http
                print(
                    f"Starting MCP HTTP transport on "
                    f"{config.mcp_http_host}:{config.mcp_http_port} "
                    f"(POST /mcp, GET /mcp/health, GET /mcp/sse)"
                )
                run_mcp_http(
                    mcp_server,
                    host=config.mcp_http_host,
                    port=config.mcp_http_port,
                )
            else:
                mcp_server.serve()
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
        # Close vector store (always Qdrant; close() is a no-op there)
        if hasattr(semantic_search, '_store') and hasattr(semantic_search._store, 'close'):
            semantic_search._store.close()


if __name__ == "__main__":
    main()
