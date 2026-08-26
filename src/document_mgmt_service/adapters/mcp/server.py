from __future__ import annotations

import base64
import json
import logging
import sys
from typing import Any

from document_mgmt_service.application.ingestion import IngestionError, IngestionService
from document_mgmt_service.application.health import HealthService
from document_mgmt_service.application.access import (
    AccessDeniedError,
    AccessIntent,
    ApprovalRequiredError,
    DocumentAccessService,
)
from document_mgmt_service.config import AppConfig
from document_mgmt_service.domain.models import DocumentPrivacyClassification


class MCPServer:
    def __init__(
        self,
        *,
        config: AppConfig,
        health_service: HealthService,
        ingestion_service: IngestionService,
        access_service: DocumentAccessService,
    ) -> None:
        self._config = config
        self._health_service = health_service
        self._ingestion_service = ingestion_service
        self._access_service = access_service
        self._logger = logging.getLogger("document_mgmt_service.mcp")

    def serve(self) -> None:
        reader = sys.stdin.buffer
        writer = sys.stdout.buffer
        while True:
            message = self._read_message(reader)
            if message is None:
                return
            response = self._dispatch(message)
            if response is not None:
                self._write_message(writer, response)

    def _dispatch(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")

        try:
            if method == "initialize":
                return self._result(
                    request_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": self._config.mcp_server_name,
                            "version": self._config.mcp_server_version,
                        },
                        "capabilities": {"tools": {"listChanged": False}},
                    },
                )
            if method == "tools/list":
                return self._result(
                    request_id,
                    {
                        "tools": [
                            self._tool_spec(
                                "upload_document",
                                "Ingest a PDF or image document from base64 content.",
                            ),
                            self._tool_spec(
                                "get_document_metadata",
                                "Return the processing status for a document version.",
                            ),
                            self._tool_spec(
                                "get_document_description",
                                "Return the description for a document version.",
                            ),
                            self._tool_spec(
                                "list_documents",
                                "List known documents with summaries and provenance.",
                            ),
                            self._tool_spec(
                                "search_documents",
                                "Perform document-level semantic search with policy enforcement.",
                            ),
                            self._tool_spec(
                                "search_document_content",
                                "Perform content-level semantic search with policy enforcement.",
                            ),
                            self._tool_spec(
                                "get_evidence",
                                "Retrieve supporting evidence for a document version.",
                            ),
                            self._tool_spec(
                                "get_page",
                                "Retrieve a specific page from a document with policy enforcement.",
                            ),
                            self._tool_spec(
                                "get_document",
                                "Retrieve a whole document from original storage when policy allows.",
                            ),
                            self._tool_spec(
                                "request_sensitive_access",
                                "Request approval for sensitive access without bypassing policy.",
                            ),
                            self._tool_spec(
                                "get_document_ocr",
                                "Return OCR-extracted text and metadata for a document version.",
                            ),
                            self._tool_spec(
                                "get_document_image",
                                "Return raw document image/file bytes for a document version.",
                            ),
                        ]
                    },
                )
            if method == "tools/call":
                return self._handle_tool_call(request_id, message.get("params", {}))
            if method == "ping":
                return self._result(request_id, {"pong": True})
            if request_id is not None:
                return self._error(request_id, -32601, f"Method not found: {method}")
            return None
        except Exception as exc:  # pragma: no cover - defensive server guard
            self._logger.exception("MCP request failed")
            if request_id is not None:
                return self._error(request_id, -32603, str(exc))
            return None

    def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name == "upload_document":
            payload = self._ingest_document(arguments)
            return self._tool_result(request_id, payload)

        if name == "get_document_metadata":
            document_id = arguments.get("document_id")
            version = int(arguments.get("version"))
            status = self._access_service.get_document_metadata(str(document_id), version)
            return self._tool_result(
                request_id,
                status.payload,
            )

        if name == "get_document_description":
            document_id = arguments.get("document_id")
            version = int(arguments.get("version"))
            response = self._access_service.get_document_description(str(document_id), version)
            return self._tool_result(request_id, response.payload)

        if name == "list_documents":
            return self._tool_result(request_id, self._access_service.list_documents())

        if name == "search_documents":
            query = str(arguments.get("query") or "")
            if not query:
                raise IngestionError("query is required")
            result = self._access_service.search_documents(
                query,
                limit=int(arguments.get("limit", 10)),
                requestor=str(arguments.get("requestor")) if arguments.get("requestor") else None,
            )
            return self._tool_result(
                request_id,
                result,
            )

        if name == "search_document_content":
            query = str(arguments.get("query") or "")
            if not query:
                raise IngestionError("query is required")
            result = self._access_service.search_document_content(
                query,
                limit=int(arguments.get("limit", 10)),
                document_id=arguments.get("document_id"),
                version=int(arguments["version"]) if arguments.get("version") is not None else None,
                requestor=str(arguments.get("requestor")) if arguments.get("requestor") else None,
            )
            return self._tool_result(
                request_id,
                result,
            )

        if name == "get_evidence":
            query = str(arguments.get("query") or "")
            if not query:
                raise IngestionError("query is required")
            document_id = str(arguments.get("document_id") or "")
            version = int(arguments.get("version"))
            if not document_id or not query:
                raise IngestionError("document_id and query are required")
            result = self._access_service.get_evidence(
                document_id=document_id,
                version=version,
                query=query,
                limit=int(arguments.get("limit", 5)),
                requestor=str(arguments.get("requestor")) if arguments.get("requestor") else None,
            )
            return self._tool_result(
                request_id,
                result,
            )

        if name == "get_page":
            document_id = str(arguments.get("document_id") or "")
            version = int(arguments.get("version"))
            page_number = int(arguments.get("page_number"))
            try:
                page_response = self._access_service.get_page(
                    document_id,
                    version,
                    page_number,
                    requestor=str(arguments.get("requestor")) if arguments.get("requestor") else None,
                )
                return self._tool_result(request_id, page_response.payload)
            except ApprovalRequiredError as exc:
                return self._tool_result(request_id, {"access_action": exc.decision.action.value, "access_reason": exc.decision.reason})
            except AccessDeniedError as exc:
                return self._tool_result(request_id, {"access_action": exc.decision.action.value, "access_reason": exc.decision.reason})
            except KeyError as exc:
                return self._error(request_id, -32602, str(exc))

        if name == "get_document":
            document_id = str(arguments.get("document_id") or "")
            version = int(arguments.get("version"))
            if not document_id:
                raise IngestionError("document_id is required")
            try:
                response = self._access_service.get_document(
                    document_id,
                    version,
                    requestor=str(arguments.get("requestor")) if arguments.get("requestor") else None,
                )
                return self._tool_result(request_id, response.payload)
            except ApprovalRequiredError as exc:
                return self._tool_result(request_id, {"access_action": exc.decision.action.value, "access_reason": exc.decision.reason, "content_base64": None})
            except AccessDeniedError as exc:
                return self._tool_result(request_id, {"access_action": exc.decision.action.value, "access_reason": exc.decision.reason, "content_base64": None})
            except KeyError as exc:
                return self._error(request_id, -32602, str(exc))

        if name == "request_sensitive_access":
            document_id = str(arguments.get("document_id") or "")
            version = int(arguments.get("version"))
            intent = AccessIntent(str(arguments.get("intent") or AccessIntent.WHOLE_DOCUMENT.value))
            query = arguments.get("query")
            return self._tool_result(
                request_id,
                self._access_service.request_sensitive_access(
                    document_id=document_id,
                    version=version,
                    intent=intent,
                    query=str(query) if query is not None else None,
                    requestor=str(arguments.get("requestor")) if arguments.get("requestor") else None,
                ),
            )

        if name == "get_document_ocr":
            document_id = str(arguments.get("document_id") or "")
            version = int(arguments.get("version"))
            try:
                record = self._access_service._load_record(document_id, version)
            except KeyError as exc:
                return self._error(request_id, -32602, str(exc))
            return self._tool_result(
                request_id,
                {
                    "document_id": record.document_id,
                    "version": record.version,
                    "processing_status": record.processing_status.value,
                    "extracted_text": record.extracted_text,
                    "extracted_text_excerpt": record.extracted_text_excerpt,
                    "file_kind": record.file_kind.value,
                    "original_filename": record.original_filename,
                    "content_type": record.content_type,
                    "metadata": record.metadata,
                    "description": record.description,
                    "privacy": record.privacy.value,
                },
            )

        if name == "get_document_image":
            document_id = str(arguments.get("document_id") or "")
            version = int(arguments.get("version"))
            try:
                response = self._access_service.get_document(
                    document_id,
                    version,
                    requestor=str(arguments.get("requestor")) if arguments.get("requestor") else None,
                )
                return self._tool_result(request_id, {
                    "document_id": document_id,
                    "version": version,
                    "filename": response.payload.get("filename"),
                    "content_type": response.payload.get("content_type"),
                    "storage_key": response.payload.get("storage_key"),
                    "content_base64": response.payload.get("content_base64"),
                    "access_action": response.decision.action.value,
                    "access_reason": response.decision.reason,
                })
            except ApprovalRequiredError as exc:
                return self._tool_result(request_id, {"access_action": exc.decision.action.value, "access_reason": exc.decision.reason, "content_base64": None})
            except AccessDeniedError as exc:
                return self._tool_result(request_id, {"access_action": exc.decision.action.value, "access_reason": exc.decision.reason, "content_base64": None})
            except KeyError as exc:
                return self._error(request_id, -32602, str(exc))

        return self._error(request_id, -32602, f"Unknown tool: {name}")

    def _ingest_document(self, arguments: dict[str, Any]) -> dict[str, Any]:
        filename = arguments.get("filename")
        content_base64 = arguments.get("content_base64")
        if not filename or not content_base64:
            raise IngestionError("filename and content_base64 are required")
        content = base64.b64decode(content_base64)
        metadata = arguments.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        result = self._ingestion_service.ingest(
            request=self._ingestion_service_request(
                filename=str(filename),
                content=content,
                content_type=arguments.get("content_type"),
                document_id=arguments.get("document_id"),
                description=arguments.get("description"),
                privacy_hint=self._parse_privacy(arguments.get("privacy")),
                metadata=metadata,
            )
        )
        return {
            "document_id": result.document_id,
            "version": result.version,
            "processing_status": result.processing_status.value,
            "privacy": result.privacy.value,
            "description": result.description,
            "metadata": result.metadata,
            "storage_key": result.storage_key,
            "sha256": result.sha256,
            "semantic_index_status": result.semantic_index_status.value,
            "chunk_count": result.chunk_count,
        }

    def _ingestion_service_request(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None,
        document_id: str | None,
        description: str | None,
        privacy_hint: DocumentPrivacyClassification | None,
        metadata: dict[str, Any],
    ):
        from document_mgmt_service.domain.models import DocumentIngestionRequest

        return DocumentIngestionRequest(
            original_filename=filename,
            content=content,
            content_type=content_type,
            document_id=document_id,
            description_hint=description,
            privacy_hint=privacy_hint,
            metadata=metadata,
        )

    def _tool_spec(self, name: str, description: str) -> dict[str, Any]:
        if name == "upload_document":
            schema = {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content_base64": {"type": "string"},
                    "content_type": {"type": "string"},
                    "document_id": {"type": "string"},
                    "privacy": {"type": "string"},
                    "description": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["filename", "content_base64"],
                "additionalProperties": False,
            }
        elif name in {"get_document_metadata", "get_document_description"}:
            schema = {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                },
                "required": ["document_id", "version"],
                "additionalProperties": False,
            }
        elif name == "list_documents":
            schema = {"type": "object", "properties": {}, "additionalProperties": False}
        elif name == "search_documents":
            schema = {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "privacy": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        elif name == "search_document_content":
            schema = {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "privacy": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        elif name == "get_evidence":
            schema = {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["document_id", "version", "query"],
                "additionalProperties": False,
            }
        elif name == "get_page":
            schema = {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "page_number": {"type": "integer"},
                },
                "required": ["document_id", "version", "page_number"],
                "additionalProperties": False,
            }
        elif name == "get_document":
            schema = {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                },
                "required": ["document_id", "version"],
                "additionalProperties": False,
            }
        elif name == "request_sensitive_access":
            schema = {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "intent": {"type": "string"},
                    "query": {"type": "string"},
                    "requestor": {"type": "string"},
                },
                "required": ["document_id", "version", "intent"],
                "additionalProperties": False,
            }
        elif name == "get_document_ocr":
            schema = {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                },
                "required": ["document_id", "version"],
                "additionalProperties": False,
            }
        elif name == "get_document_image":
            schema = {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "requestor": {"type": "string"},
                },
                "required": ["document_id", "version"],
                "additionalProperties": False,
            }
        else:
            schema = {"type": "object", "properties": {}, "additionalProperties": False}
        return {"name": name, "description": description, "inputSchema": schema}

    def _tool_result(self, request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return self._result(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload, default=str, separators=(",", ":")),
                    }
                ],
                "isError": False,
            },
        )

    def _result(self, request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _parse_privacy(self, value: Any) -> DocumentPrivacyClassification | None:
        if value in {None, ""}:
            return None
        return DocumentPrivacyClassification(str(value))

    def _read_message(self, reader: Any) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            line = reader.readline()
            if not line:
                return None
            line = line.decode("utf-8").rstrip("\r\n")
            if not line:
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            return None
        payload = reader.read(content_length)
        return json.loads(payload.decode("utf-8"))

    def _write_message(self, writer: Any, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
        writer.write(header)
        writer.write(payload)
        writer.flush()
