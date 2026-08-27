"""HTTP server adapter for the Validator Service.

Exposes REST endpoints for document validation, template management,
and alert retrieval.
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from validator_service.application.validation import ValidationService
from validator_service.domain.models import ValidationRequest

logger = logging.getLogger("validator_service.http")


class ValidatorHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the validator service."""

    validation_service: ValidationService

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    def do_GET(self) -> None:
        path = self.path.rstrip("/")

        if path == "/healthz":
            return self._json_response(self.validation_service.health())

        if path == "/templates":
            registry = self.validation_service._registry
            templates = registry.list_all()
            return self._json_response({
                "templates": [
                    {
                        "template_id": t.template_id,
                        "document_type": t.document_type,
                        "document_sub_type": t.document_sub_type,
                        "variant_name": t.variant_name,
                        "description": t.description,
                        "field_count": len(t.field_rules),
                    }
                    for t in templates
                ],
                "count": len(templates),
            })

        if path.startswith("/templates/"):
            template_id = path.split("/")[-1]
            registry = self.validation_service._registry
            template = registry.get_template(template_id)
            if not template:
                return self._json_response({"error": f"Template not found: {template_id}"}, 404)
            return self._json_response({
                "template_id": template.template_id,
                "document_type": template.document_type,
                "document_sub_type": template.document_sub_type,
                "variant_name": template.variant_name,
                "description": template.description,
                "field_rules": [
                    {"name": fr.name, "type": fr.field_type.value, "required": fr.required}
                    for fr in template.field_rules
                ],
            })

        if path == "/alerts":
            alert_sender = self.validation_service._alert_sender
            if not alert_sender or not hasattr(alert_sender, "get_alerts"):
                return self._json_response({"alerts": [], "count": 0})
            alerts = alert_sender.get_alerts()
            return self._json_response({"alerts": alerts, "count": len(alerts)})

        self._json_response({"error": f"Not found: {path}"}, 404)

    def do_POST(self) -> None:
        path = self.path.rstrip("/")

        if path == "/validate":
            return self._handle_validate()
        if path == "/validate/image":
            return self._handle_validate_image()

        self._json_response({"error": f"Not found: {path}"}, 404)

    def _handle_validate(self) -> None:
        """Validate a document by ID."""
        body = self._read_body()
        if body is None:
            return

        request = ValidationRequest(
            document_id=body.get("document_id", ""),
            version=int(body.get("version", 1)),
            document_type=body.get("document_type"),
            document_sub_type=body.get("document_sub_type"),
            extracted_text=body.get("extracted_text"),
            metadata=body.get("metadata") or {},
            request_id=body.get("request_id"),
        )
        result = self.validation_service.validate(request)
        return self._json_response(self._result_to_dict(result))

    def _handle_validate_image(self) -> None:
        """Validate a document image directly."""
        body = self._read_body()
        if body is None:
            return

        import base64
        content_b64 = body.get("content_base64")
        if not content_b64:
            return self._json_response({"error": "content_base64 is required"}, 400)

        image_bytes = base64.b64decode(content_b64)
        request = ValidationRequest(
            document_id=body.get("document_id", "direct-upload"),
            version=int(body.get("version", 1)),
            document_type=body.get("document_type"),
            document_sub_type=body.get("document_sub_type"),
            extracted_text=body.get("extracted_text"),
            metadata=body.get("metadata") or {},
            image_bytes=image_bytes,
            image_mime_type=body.get("image_mime_type", "image/jpeg"),
            request_id=body.get("request_id"),
        )
        result = self.validation_service.validate(request)
        return self._json_response(self._result_to_dict(result))

    def _result_to_dict(self, result: Any) -> dict[str, Any]:
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
                    "days_until_expiry": tc.days_until_expiry,
                    "details": tc.details,
                }
                for tc in result.temporal_checks
            ],
            "steps": [
                {"name": s.step_name, "passed": s.passed, "details": s.details}
                for s in result.steps
            ],
            "alerts": [
                {"severity": a.severity.value, "type": a.alert_type, "message": a.message}
                for a in result.alerts
            ],
            "notes": result.notes,
            "latency_ms": result.latency_ms,
        }

    def _read_body(self) -> dict[str, Any] | None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self._json_response({"error": "Empty request body"}, 400)
            return None
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
            return None

    def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, default=str, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_http_server(
    validation_service: ValidationService,
    port: int = 8083,
) -> None:
    """Start the HTTP server."""
    handler = type(
        "Handler",
        (ValidatorHTTPHandler,),
        {"validation_service": validation_service},
    )
    server = HTTPServer(("0.0.0.0", port), handler)
    logger.info("Validator HTTP server listening on port %d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down validator HTTP server")
        server.shutdown()
