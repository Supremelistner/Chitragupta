"""Infrastructure — model-based document validator adapter.

Calls the model service to visually inspect a document against a template.
The model checks structural consistency, authenticity indicators, and
whether the document matches the expected template.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from validator_service.domain.models import (
    AlertPayload,
    AlertSeverity,
    DocumentTemplate,
    FieldType,
    FieldValidation,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    ValidationStep,
)

logger = logging.getLogger("validator_service.model_validator")


# ---------------------------------------------------------------------------
# Validation prompt — ask the model to check document authenticity
# ---------------------------------------------------------------------------

_VALIDATION_PROMPT = """You are a document validation specialist for an Indian document processing system.

TASK: Examine the document image and verify its authenticity and structural consistency.

TEMPLATE CONTEXT:
{template_context}

OUTPUT RULES:
- Respond with ONLY a valid JSON object. No markdown, no explanation.
- If you cannot validate, return: {{"error": true, "message": "<reason>"}}

JSON SCHEMA:
{{
  "is_authentic": true/false,
  "confidence": <0.0-1.0>,
  "risk_score": <0.0-1.0, where 1.0 = definitely fake>,
  "matched_template_variant": "<which variant this looks like, or 'unknown'>",
  "structural_checks": [
    {{
      "check_name": "<what was checked>",
      "passed": true/false,
      "details": "<what was found>",
      "severity": "info | warning | critical"
    }}
  ],
  "authenticity_indicators": [
    {{
      "indicator": "<e.g. 'consistent_font', 'valid_format', 'government_watermark'>",
      "present": true/false,
      "details": "<observation>"
    }}
  ],
  "fraud_indicators": [
    {{
      "indicator": "<e.g. 'misaligned_text', 'inconsistent_photo', 'altered_number'>",
      "details": "<what looks wrong>"
    }}
  ],
  "notes": ["<any additional observations>"]
}}

VALIDATION RULES:
- Check if the document has the visual structure of a real {doc_type} ({variant}).
- For identity documents: verify photo looks natural, text is properly aligned,
  security features are present (hologram, watermark, microprint if visible).
- For academic documents: verify layout matches official format, marks/grades
  are in expected columns, board/school name is prominent.
- Check for signs of tampering: mismatched fonts, inconsistent spacing,
  digital artifacts, blurry sections, re-composited elements.
- risk_score > 0.5 = SUSPICIOUS, > 0.8 = likely FAKE.
- is_authentic = false triggers an alert to the document service.
- DO NOT mark a document as fake unless you see clear indicators.
"""


class ModelValidationAdapter:
    """Uses the model service to visually validate documents.

    This adapter wraps an HTTP client to the model service and sends
    the document image along with template context for visual validation.
    """

    def __init__(
        self,
        *,
        model_service_url: str = "http://localhost:8081",
        timeout_seconds: int = 30,
    ) -> None:
        self._model_service_url = model_service_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def ping(self) -> None:
        """Check that the model service is reachable."""
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{self._model_service_url}/healthz",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Model service health check failed: {resp.status}")
        except Exception as exc:
            raise RuntimeError(f"Model service unreachable: {exc}")

    def validate_document(
        self,
        request: ValidationRequest,
        template: DocumentTemplate | None = None,
    ) -> ValidationResult:
        """Send document to model service for visual validation."""
        start = time.monotonic()

        if not request.image_bytes:
            return ValidationResult(
                document_id=request.document_id,
                version=request.version,
                status=ValidationStatus.ERROR,
                notes=("No image provided for model validation",),
                latency_ms=(time.monotonic() - start) * 1000,
                request_id=request.request_id,
                created_at=datetime.now(timezone.utc),
            )

        # Build template context for the prompt
        template_context = self._build_template_context(template) if template else "No template provided."

        # Build the prompt
        doc_type = template.document_sub_type if template else "unknown"
        variant = template.variant_name if template else "unknown"
        prompt = _VALIDATION_PROMPT.format(
            template_context=template_context,
            doc_type=doc_type,
            variant=variant,
        )

        # Call the model service inference endpoint
        try:
            model_output = self._call_model_service(
                request.image_bytes,
                request.image_mime_type or "image/jpeg",
                prompt,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.exception("Model validation call failed")
            return ValidationResult(
                document_id=request.document_id,
                version=request.version,
                status=ValidationStatus.ERROR,
                notes=(f"Model service error: {exc}",),
                latency_ms=latency_ms,
                request_id=request.request_id,
                created_at=datetime.now(timezone.utc),
            )

        # Parse model output
        try:
            parsed = json.loads(model_output)
        except json.JSONDecodeError:
            latency_ms = (time.monotonic() - start) * 1000
            return ValidationResult(
                document_id=request.document_id,
                version=request.version,
                status=ValidationStatus.ERROR,
                notes=("Model returned invalid JSON", f"Raw: {model_output[:500]}"),
                latency_ms=latency_ms,
                request_id=request.request_id,
                created_at=datetime.now(timezone.utc),
            )

        # Build result from model output
        latency_ms = (time.monotonic() - start) * 1000
        is_authentic = parsed.get("is_authentic", True)
        risk_score = float(parsed.get("risk_score", 0.0))
        confidence = float(parsed.get("confidence", 0.0))

        # Build validation steps
        steps: list[ValidationStep] = []
        for check in parsed.get("structural_checks", []):
            steps.append(ValidationStep(
                step_name=f"model_{check.get('check_name', 'unknown')}",
                passed=check.get("passed", True),
                details=check.get("details", ""),
                confidence=confidence,
            ))

        # Determine status
        if risk_score >= 0.7:
            status = ValidationStatus.INVALID
        elif risk_score >= 0.4:
            status = ValidationStatus.SUSPICIOUS
        else:
            status = ValidationStatus.VALID

        # Build alerts
        alerts: list[AlertPayload] = []
        if not is_authentic or risk_score >= 0.5:
            severity = AlertSeverity.CRITICAL if risk_score >= 0.7 else AlertSeverity.WARNING
            alerts.append(AlertPayload(
                document_id=request.document_id,
                version=request.version,
                severity=severity,
                alert_type="fake_document" if not is_authentic else "high_risk",
                message=f"Model validation: risk_score={risk_score:.2f}, authentic={is_authentic}",
                details={
                    "fraud_indicators": parsed.get("fraud_indicators", []),
                    "matched_variant": parsed.get("matched_template_variant"),
                },
                risk_score=risk_score,
                request_id=request.request_id,
                created_at=datetime.now(timezone.utc),
            ))

        return ValidationResult(
            document_id=request.document_id,
            version=request.version,
            status=status,
            matched_template=template.template_id if template else None,
            steps=tuple(steps),
            overall_confidence=confidence,
            is_authentic=is_authentic,
            risk_score=risk_score,
            alerts=tuple(alerts),
            notes=tuple(parsed.get("notes", [])),
            latency_ms=latency_ms,
            request_id=request.request_id,
            created_at=datetime.now(timezone.utc),
        )

    def _build_template_context(self, template: DocumentTemplate) -> str:
        """Build a human-readable template description for the model prompt."""
        lines = [
            f"Document type: {template.document_type} / {template.document_sub_type}",
            f"Variant: {template.variant_name}",
            f"Description: {template.description}",
            f"Expected fields ({len(template.field_rules)}):",
        ]
        for fr in template.field_rules:
            req = "REQUIRED" if fr.required else "optional"
            lines.append(f"  - {fr.name} ({fr.field_type.value}, {req}): {fr.description}")
            if fr.regex:
                lines.append(f"    Pattern: {fr.regex}")
            if fr.allowed_values:
                lines.append(f"    Allowed: {', '.join(fr.allowed_values)}")

        if template.structural_rules:
            lines.append(f"Structural rules ({len(template.structural_rules)}):")
            for sr in template.structural_rules:
                lines.append(f"  - {sr.name}: {sr.description}")

        lines.append(f"Pages: {template.min_pages}-{template.max_pages}")
        return "\n".join(lines)

    def _call_model_service(
        self,
        image_bytes: bytes,
        image_mime_type: str,
        prompt: str,
    ) -> str:
        """Call the model service's inference endpoint."""
        import base64
        import urllib.request

        b64_image = base64.b64encode(image_bytes).decode("ascii")

        payload = json.dumps({
            "task": "custom",
            "image_base64": b64_image,
            "image_mime_type": image_mime_type,
            "prompt": prompt,
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 2048,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._model_service_url}/infer",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("output", "")

    def close(self) -> None:
        pass
