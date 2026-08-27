"""Validator Service — main entrypoint.

Wires up the template registry, model validator, alert sender,
and starts both MCP (stdio) and HTTP servers.

Usage:
    python -m validator_service            # MCP on stdin/stdout
    python -m validator_service --http     # HTTP on configured port
    python -m validator_service --both     # Both (HTTP in background)
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from validator_service.config import (
    VALIDATOR_SERVICE_HTTP_PORT,
    MODEL_SERVICE_URL,
    DOCUMENT_SERVICE_URL,
)
from validator_service.domain.ports import (
    AlertSender,
    DocumentServiceClient,
    ModelValidator,
    TemplateRegistry,
)
from validator_service.application.validation import ValidationService
from validator_service.infrastructure.templates import InMemoryTemplateRegistry
from validator_service.infrastructure.model_validator import ModelValidationAdapter
from validator_service.infrastructure.alert import DocumentServiceAlertAdapter
from validator_service.infrastructure.document_client import HTTPDocumentServiceClient
from validator_service.adapters.mcp.server import MCPServer
from validator_service.adapters.http.server import run_http_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("validator_service")


def build_service() -> ValidationService:
    """Wire up all components. Connections are lazy — HTTP server starts immediately."""
    # Template registry (always available — in-memory)
    registry: TemplateRegistry = InMemoryTemplateRegistry()
    logger.info("Template registry loaded with %d templates", len(registry.list_all()))

    # Create adapters without pinging — they connect on first use
    model_validator: ModelValidator = ModelValidationAdapter(
        model_service_url=MODEL_SERVICE_URL,
    )
    alert_sender: AlertSender = DocumentServiceAlertAdapter(
        document_service_url=DOCUMENT_SERVICE_URL,
    )
    doc_client: DocumentServiceClient = HTTPDocumentServiceClient(
        document_service_url=DOCUMENT_SERVICE_URL,
    )

    logger.info(
        "Adapters created (model=%s, document=%s) — connections are lazy",
        MODEL_SERVICE_URL, DOCUMENT_SERVICE_URL,
    )

    return ValidationService(
        template_registry=registry,
        model_validator=model_validator,
        alert_sender=alert_sender,
        document_client=doc_client,
    )


def _probe_connections() -> None:
    """Try to connect to dependent services in background (non-blocking)."""
    import threading

    def _probe() -> None:
        time.sleep(2)  # Give other services a moment to start
        services = [
            ("model_validator", ModelValidationAdapter(model_service_url=MODEL_SERVICE_URL)),
            ("alert_sender", DocumentServiceAlertAdapter(document_service_url=DOCUMENT_SERVICE_URL)),
            ("document_client", HTTPDocumentServiceClient(document_service_url=DOCUMENT_SERVICE_URL)),
        ]
        for name, adapter in services:
            try:
                adapter.ping()
                logger.info("%s connected", name)
            except Exception as exc:
                logger.warning("%s not available yet: %s", name, exc)

    threading.Thread(target=_probe, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validator Service")
    parser.add_argument("--http", action="store_true", help="Start HTTP server")
    parser.add_argument("--both", action="store_true", help="Start both MCP and HTTP servers")
    parser.add_argument("--port", type=int, default=VALIDATOR_SERVICE_HTTP_PORT, help="HTTP port")
    args = parser.parse_args()

    service = build_service()
    _probe_connections()  # Non-blocking background probe

    if args.both:
        # Start HTTP in background thread
        http_thread = threading.Thread(
            target=run_http_server,
            args=(service, args.port),
            daemon=True,
        )
        http_thread.start()
        logger.info("HTTP server started in background on port %d", args.port)
        # Run MCP on stdin/stdout
        MCPServer(service).serve()
    elif args.http:
        run_http_server(service, args.port)
    else:
        MCPServer(service).serve()


if __name__ == "__main__":
    main()
