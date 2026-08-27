"""MCP server adapter for the Validator Service.

Exposes document validation tools via Model Context Protocol,
allowing the agent service to validate documents before accepting them.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from typing import Any

from validator_service.application.validation import ValidationService
from validator_service.domain.models import (
    ValidationRequest,
    AlertSeverity,
)

logger = logging.getLogger("validator_service.mcp")


class MCPServer:
    """MCP server exposing validator tools."""

    def __init__(self, validation_service: ValidationService) -> None:
        self._service = validation_service

    def serve(self) -> None:
        """Run the MCP server on stdin/stdout."""
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
                return self._result(request_id, {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "validator-service",
                        "version": "1.0.0",
                    },
                    "capabilities": {"tools": {"listChanged": False}},
                })
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
            logger.exception("MCP request failed")
            if request_id is not None:
                return self._error(request_id, -32603, str(exc))
            return None

    def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}

        if name == "validate_document":
            return self._validate_document(request_id, args)
        if name == "validate_document_image":
            return self._validate_document_image(request_id, args)
        if name == "list_templates":
            return self._list_templates(request_id, args)
        if name == "get_template":
            return self._get_template(request_id, args)
        if name == "get_alerts":
            return self._get_alerts(request_id, args)
        if name == "health":
            return self._tool_result(request_id, self._service.health())

        return self._error(request_id, -32602, f"Unknown tool: {name}")

    def _validate_document(self, request_id: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate a document by ID (fetches metadata from document service)."""
        document_id = args.get("document_id")
        version = args.get("version", 1)
        if not document_id:
            return self._error(request_id, -32602, "document_id is required")

        request = ValidationRequest(
            document_id=document_id,
            version=int(version),
            request_id=args.get("request_id"),
        )
        result = self._service.validate(request)
        return self._tool_result(request_id, self._result_payload(result))

    def _validate_document_image(self, request_id: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Validate a document image directly (with base64 content)."""
        content_base64 = args.get("content_base64")
        image_mime_type = args.get("image_mime_type", "image/jpeg")
        document_id = args.get("document_id", "direct-upload")
        version = args.get("version", 1)

        if not content_base64:
            return self._error(request_id, -32602, "content_base64 is required")

        image_bytes = base64.b64decode(content_base64)

        request = ValidationRequest(
            document_id=document_id,
            version=int(version),
            document_type=args.get("document_type"),
            document_sub_type=args.get("document_sub_type"),
            extracted_text=args.get("extracted_text"),
            metadata=args.get("metadata") or {},
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            request_id=args.get("request_id"),
        )
        result = self._service.validate(request)
        return self._tool_result(request_id, self._result_payload(result))

    def _list_templates(self, request_id: Any, args: dict[str, Any]) -> dict[str, Any]:
        """List all registered document templates."""
        registry = self._service._registry
        templates = registry.get_templates(
            document_type=args.get("document_type"),
            document_sub_type=args.get("document_sub_type"),
        )
        return self._tool_result(request_id, {
            "templates": [
                {
                    "template_id": t.template_id,
                    "document_type": t.document_type,
                    "document_sub_type": t.document_sub_type,
                    "variant_name": t.variant_name,
                    "description": t.description,
                    "field_count": len(t.field_rules),
                    "required_fields": [f.name for f in t.field_rules if f.required],
                    "min_pages": t.min_pages,
                    "max_pages": t.max_pages,
                }
                for t in templates
            ],
            "count": len(templates),
        })

    def _get_template(self, request_id: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Get a specific template by ID."""
        template_id = args.get("template_id")
        if not template_id:
            return self._error(request_id, -32602, "template_id is required")

        registry = self._service._registry
        template = registry.get_template(template_id)
        if not template:
            return self._error(request_id, -32602, f"Template not found: {template_id}")

        return self._tool_result(request_id, {
            "template_id": template.template_id,
            "document_type": template.document_type,
            "document_sub_type": template.document_sub_type,
            "variant_name": template.variant_name,
            "description": template.description,
            "field_rules": [
                {
                    "name": fr.name,
                    "type": fr.field_type.value,
                    "required": fr.required,
                    "description": fr.description,
                    "regex": fr.regex,
                    "allowed_values": fr.allowed_values,
                }
                for fr in template.field_rules
            ],
            "structural_rules": [
                {"name": sr.name, "description": sr.description, "type": sr.rule_type}
                for sr in template.structural_rules
            ],
            "temporal_rules": [
                {
                    "rule_type": tr.rule_type.value,
                    "description": tr.description,
                    "expiry_field": tr.expiry_field,
                    "issue_date_field": tr.issue_date_field,
                    "lifetime_days": tr.lifetime_days,
                    "warning_days": tr.warning_days,
                }
                for tr in template.temporal_rules
            ],
            "min_pages": template.min_pages,
            "max_pages": template.max_pages,
        })

    def _get_alerts(self, request_id: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Get logged validation alerts."""
        alert_sender = self._service._alert_sender
        if not alert_sender or not hasattr(alert_sender, "get_alerts"):
            return self._tool_result(request_id, {"alerts": [], "error": "Alert log not available"})

        alerts = alert_sender.get_alerts(document_id=args.get("document_id"))
        return self._tool_result(request_id, {"alerts": alerts, "count": len(alerts)})

    def _result_payload(self, result: Any) -> dict[str, Any]:
        """Convert a domain result to an MCP-friendly dict."""
        return {
            "document_id": result.document_id,
            "version": result.version,
            "status": result.status.value,
            "matched_template": result.matched_template,
            "overall_confidence": result.overall_confidence,
            "is_authentic": result.is_authentic,
            "risk_score": result.risk_score,
            "temporal_status": result.temporal_status.value,
            "temporal_tags": list(result.temporal_tags),
            "temporal_checks": [
                {
                    "rule_type": tc.rule_type.value,
                    "description": tc.description,
                    "status": tc.status.value,
                    "expiry_date": tc.expiry_date.isoformat() if tc.expiry_date else None,
                    "days_until_expiry": tc.days_until_expiry,
                    "details": tc.details,
                }
                for tc in result.temporal_checks
            ],
            "steps": [
                {
                    "name": s.step_name,
                    "passed": s.passed,
                    "details": s.details,
                    "confidence": s.confidence,
                }
                for s in result.steps
            ],
            "alerts": [
                {
                    "severity": a.severity.value,
                    "type": a.alert_type,
                    "message": a.message,
                    "risk_score": a.risk_score,
                }
                for a in result.alerts
            ],
            "notes": result.notes,
            "latency_ms": result.latency_ms,
        }

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "validate_document",
                "description": "Validate a document by ID against known templates. Checks structure and authenticity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "version": {"type": "integer"},
                        "request_id": {"type": "string"},
                    },
                    "required": ["document_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "validate_document_image",
                "description": "Validate a document image directly (base64). For real-time validation before ingestion.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content_base64": {"type": "string"},
                        "image_mime_type": {"type": "string"},
                        "document_id": {"type": "string"},
                        "version": {"type": "integer"},
                        "document_type": {"type": "string"},
                        "document_sub_type": {"type": "string"},
                        "extracted_text": {"type": "string"},
                        "metadata": {"type": "object"},
                        "request_id": {"type": "string"},
                    },
                    "required": ["content_base64"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_templates",
                "description": "List all registered document templates, optionally filtered by type.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_type": {"type": "string"},
                        "document_sub_type": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_template",
                "description": "Get detailed schema for a specific document template.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "template_id": {"type": "string"},
                    },
                    "required": ["template_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_alerts",
                "description": "Get logged validation alerts (fake/suspicious document detections).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "health",
                "description": "Check validator service health.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ]

    def _tool_result(self, request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return self._result(request_id, {
            "content": [{"type": "text", "text": json.dumps(payload, default=str, separators=(",", ":"))}],
            "isError": False,
        })

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
