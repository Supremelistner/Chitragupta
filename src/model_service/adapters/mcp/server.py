"""MCP Server adapter for the Model Service.

Exposes model inference capabilities as MCP tools for agent consumption.
Follows the same MCP protocol as the Document Management service.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from typing import Any

from model_service.application.inference import InferenceService
from model_service.config import ModelServiceConfig
from model_service.domain.models import InferenceRequest, InferenceTaskType

logger = logging.getLogger("model_service.mcp")


class MCPServer:
    """MCP server that exposes model inference tools."""

    def __init__(
        self,
        *,
        config: ModelServiceConfig,
        inference_service: InferenceService,
    ) -> None:
        self._config = config
        self._inference = inference_service
        self._logger = logger

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
                return self._result(request_id, {"tools": self._tool_specs()})
            if method == "tools/call":
                return self._handle_tool_call(request_id, message.get("params", {}))
            if method == "ping":
                return self._result(request_id, {"pong": True})
            if request_id is not None:
                return self._error(request_id, -32601, f"Method not found: {method}")
            return None
        except Exception as exc:
            self._logger.exception("MCP request failed")
            if request_id is not None:
                return self._error(request_id, -32603, str(exc))
            return None

    def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name == "classify_document":
            return self._tool_result(request_id, self._classify_document(arguments))
        if name == "verify_ocr":
            return self._tool_result(request_id, self._verify_ocr(arguments))
        if name == "extract_metadata":
            return self._tool_result(request_id, self._extract_metadata(arguments))
        if name == "summarize_content":
            return self._tool_result(request_id, self._summarize_content(arguments))
        if name == "list_providers":
            return self._tool_result(request_id, {"providers": self._inference.list_providers()})
        if name == "get_active_provider":
            return self._tool_result(request_id, {"active_provider": self._inference.get_active_provider()})
        if name == "set_active_provider":
            provider = str(arguments.get("provider") or "")
            if not provider:
                raise ValueError("provider is required")
            return self._tool_result(request_id, self._inference.set_active_provider(provider))
        if name == "health":
            return self._tool_result(request_id, self._inference.health())

        return self._error(request_id, -32602, f"Unknown tool: {name}")

    def _classify_document(self, args: dict[str, Any]) -> dict[str, Any]:
        image_bytes, mime = self._extract_image(args)
        request = InferenceRequest(
            task=InferenceTaskType.DOCUMENT_CLASSIFICATION,
            image_bytes=image_bytes,
            image_mime_type=mime,
            text=args.get("text"),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _verify_ocr(self, args: dict[str, Any]) -> dict[str, Any]:
        image_bytes, mime = self._extract_image(args)
        request = InferenceRequest(
            task=InferenceTaskType.OCR_VERIFICATION,
            image_bytes=image_bytes,
            image_mime_type=mime,
            text=args.get("ocr_text", ""),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _extract_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        image_bytes, mime = self._extract_image(args)
        request = InferenceRequest(
            task=InferenceTaskType.METADATA_EXTRACTION,
            image_bytes=image_bytes,
            image_mime_type=mime,
            text=args.get("ocr_text", ""),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _summarize_content(self, args: dict[str, Any]) -> dict[str, Any]:
        request = InferenceRequest(
            task=InferenceTaskType.CONTENT_SUMMARIZATION,
            text=args.get("text", ""),
            prompt=args.get("prompt", ""),
            request_id=args.get("request_id"),
        )
        result = self._inference.infer(request)
        return self._result_payload(result)

    def _extract_image(self, args: dict[str, Any]) -> tuple[bytes | None, str | None]:
        """Extract image bytes and MIME type from arguments."""
        image_b64 = args.get("image_base64")
        mime = args.get("image_mime_type", "image/jpeg")
        if image_b64:
            return base64.b64decode(image_b64), mime

        image_path = args.get("image_path")
        if image_path:
            from pathlib import Path
            path = Path(image_path)
            if path.exists():
                return path.read_bytes(), mime

        return None, None

    def _result_payload(self, result: Any) -> dict[str, Any]:
        return {
            "task": result.task.value,
            "provider": result.provider.value,
            "model_id": result.model_id,
            "output": result.output,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "token_usage": result.token_usage,
            "request_id": result.request_id,
        }

    # -- Tool specs --

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            self._tool_spec(
                "classify_document",
                "Classify a document using vision model. Send image bytes or path.",
                {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string", "description": "Base64-encoded image"},
                        "image_path": {"type": "string", "description": "Path to image file"},
                        "image_mime_type": {"type": "string", "default": "image/jpeg"},
                        "text": {"type": "string", "description": "Additional text context"},
                        "prompt": {"type": "string", "description": "Custom prompt override"},
                    },
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "verify_ocr",
                "Verify OCR output against the original image for accuracy.",
                {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string"},
                        "image_path": {"type": "string"},
                        "image_mime_type": {"type": "string", "default": "image/jpeg"},
                        "ocr_text": {"type": "string", "description": "OCR text to verify"},
                        "prompt": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "extract_metadata",
                "Extract structured metadata from a document image.",
                {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string"},
                        "image_path": {"type": "string"},
                        "image_mime_type": {"type": "string", "default": "image/jpeg"},
                        "ocr_text": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "summarize_content",
                "Summarize document text content.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarize"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "list_providers", "List available model providers and their status.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._tool_spec(
                "get_active_provider", "Get the currently active model provider.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._tool_spec(
                "set_active_provider",
                "Switch the active model provider.",
                {
                    "type": "object",
                    "properties": {"provider": {"type": "string"}},
                    "required": ["provider"],
                    "additionalProperties": False,
                },
            ),
            self._tool_spec(
                "health", "Health check for the model service.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
        ]

    def _tool_spec(self, name: str, description: str, schema: dict) -> dict[str, Any]:
        return {"name": name, "description": description, "inputSchema": schema}

    def _tool_result(self, request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return self._result(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, default=str, separators=(",", ":"))}],
                "isError": False,
            },
        )

    def _result(self, request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

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
