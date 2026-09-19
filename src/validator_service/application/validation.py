"""Application layer — validation orchestration.

Coordinates template lookup, model-based validation, and alert dispatch.
No infrastructure knowledge — only port protocols.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from validator_service.domain.models import (
    AlertPayload,
    AlertResult,
    AlertSeverity,
    DocumentTemplate,
    FieldType,
    FieldValidation,
    StructuralRule,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    ValidationStep,
)
from validator_service.domain.ports import (
    AlertSender,
    DocumentServiceClient,
    ModelValidator,
    TemplateRegistry,
)
from validator_service import config

logger = logging.getLogger("validator_service.application")


class ValidationError(Exception):
    """Raised when validation cannot proceed."""


class ValidationService:
    """Orchestrates document validation against known templates.

    Flow:
    1. Determine document type (from pre-classified metadata or model inference)
    2. Look up matching templates from registry
    3. Run rule-based structural validation (field presence, regex, etc.)
    4. Run model-based visual validation (authenticity check)
    5. Combine results, compute risk score
    6. If suspicious/fake, send alert to document service
    """

    def __init__(
        self,
        *,
        template_registry: TemplateRegistry,
        model_validator: ModelValidator | None = None,
        alert_sender: AlertSender | None = None,
        document_client: DocumentServiceClient | None = None,
    ) -> None:
        self._registry = template_registry
        self._model_validator = model_validator
        self._alert_sender = alert_sender
        self._document_client = document_client

    def validate(self, request: ValidationRequest) -> ValidationResult:
        """Run the full validation pipeline."""
        start = time.monotonic()

        try:
            result = self._run_validation(request)
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.exception("Validation failed for %s", request.document_id)
            return ValidationResult(
                document_id=request.document_id,
                version=request.version,
                status=ValidationStatus.ERROR,
                notes=(f"Validation error: {exc}",),
                latency_ms=latency_ms,
                request_id=request.request_id,
                created_at=datetime.now(timezone.utc),
            )

        result = ValidationResult(
            document_id=result.document_id,
            version=result.version,
            status=result.status,
            matched_template=result.matched_template,
            steps=result.steps,
            overall_confidence=result.overall_confidence,
            is_authentic=result.is_authentic,
            risk_score=result.risk_score,
            alerts=result.alerts,
            notes=result.notes,
            latency_ms=(time.monotonic() - start) * 1000,
            request_id=request.request_id,
            created_at=datetime.now(timezone.utc),
        )

        # Dispatch alerts if needed
        if result.risk_score > config.ALERT_RISK_THRESHOLD or result.status in (ValidationStatus.INVALID, ValidationStatus.SUSPICIOUS):
            self._dispatch_alerts(result, request)

        return result

    def _run_validation(self, request: ValidationRequest) -> ValidationResult:
        """Core validation logic."""
        # Step 1: Enrich request if we have a document client
        doc_type = request.document_type
        doc_sub_type = request.document_sub_type
        metadata = dict(request.metadata)

        if self._document_client and not doc_type:
            try:
                doc_meta = self._document_client.get_document_metadata(
                    request.document_id, request.version,
                )
                doc_type = doc_meta.get("document_type")
                doc_sub_type = doc_meta.get("document_sub_type")
                metadata.update(doc_meta.get("model_extraction") or {})
            except Exception as exc:
                logger.warning("Could not fetch document metadata: %s", exc)

        # Step 2: Look up matching templates
        templates = self._registry.get_templates(doc_type, doc_sub_type)
        if not templates:
            # Try broader search
            templates = self._registry.get_templates(doc_type)

        if not templates:
            alert = AlertPayload(
                document_id=request.document_id,
                version=request.version,
                severity=AlertSeverity.WARNING,
                alert_type="template_unknown",
                message=f"No templates found for document type '{doc_type}' / '{doc_sub_type}'",
                details={"document_type": doc_type, "document_sub_type": doc_sub_type},
                request_id=request.request_id,
                created_at=datetime.now(timezone.utc),
            )
            return ValidationResult(
                document_id=request.document_id,
                version=request.version,
                status=ValidationStatus.UNKNOWN_DOCUMENT,
                alerts=(alert,),
                notes=(f"No templates registered for {doc_type}/{doc_sub_type}",),
            )

        # Step 3: Rule-based structural validation against best matching template
        best_template = self._pick_best_template(templates, metadata)
        structural_result = self._validate_structural(request, best_template, metadata)

        # Step 4: Model-based visual validation (if available)
        model_result = None
        if self._model_validator:
            try:
                model_result = self._model_validator.validate_document(request, best_template)
            except Exception as exc:
                logger.warning("Model validation failed: %s", exc)

        # Step 5: Combine results
        return self._combine_results(
            request, best_template, structural_result, model_result, metadata,
        )

    def _pick_best_template(
        self, templates: list[DocumentTemplate], metadata: dict[str, Any],
    ) -> DocumentTemplate:
        """Pick the best matching template from candidates.

        Heuristic: prefer templates whose expected field count is closest
        to the number of fields actually extracted.
        """
        if len(templates) == 1:
            return templates[0]

        extracted_fields = len(metadata.get("fields", {}))

        best = templates[0]
        best_score = -1
        for t in templates:
            required_fields = sum(1 for f in t.field_rules if f.required)
            # Score = overlap of expected fields with extracted fields
            score = min(required_fields, extracted_fields) / max(required_fields, 1)
            if score > best_score:
                best_score = score
                best = t

        return best

    def _validate_structural(
        self,
        request: ValidationRequest,
        template: DocumentTemplate,
        metadata: dict[str, Any],
    ) -> ValidationStep:
        """Rule-based validation: check field presence, types, constraints."""
        extracted_fields = metadata.get("fields", {})
        violations: list[str] = []
        field_results: list[FieldValidation] = []

        # Case-insensitive field lookup: the model may emit keys like
        # "aadhaar_number" while a template uses "Aadhaar_number". Without
        # this, the strict lookup misses and every required field is flagged
        # as missing even though the data is present.
        fields_ci = {k.casefold(): v for k, v in extracted_fields.items()}
        for rule in template.field_rules:
            field_data = fields_ci.get(rule.name.casefold())
            present = field_data is not None and field_data.get("value") is not None
            value = field_data.get("value") if field_data else None
            field_violations: list[str] = []

            if rule.required and not present:
                field_violations.append(f"Required field '{rule.name}' is missing")

            if present and value:
                # Regex check
                if rule.regex:
                    import re
                    if not re.search(rule.regex, str(value)):
                        field_violations.append(
                            f"Value '{value}' does not match pattern '{rule.regex}'"
                        )

                # Length checks
                if rule.min_length is not None and len(str(value)) < rule.min_length:
                    field_violations.append(
                        f"Value too short ({len(str(value))} < {rule.min_length})"
                    )
                if rule.max_length is not None and len(str(value)) > rule.max_length:
                    field_violations.append(
                        f"Value too long ({len(str(value))} > {rule.max_length})"
                    )

                # Allowed values check
                if rule.allowed_values and str(value) not in rule.allowed_values:
                    field_violations.append(
                        f"Value '{value}' not in allowed values {rule.allowed_values}"
                    )

            violations.extend(field_violations)

            field_results.append(FieldValidation(
                field_name=rule.name,
                expected_type=rule.field_type,
                present=present,
                value=value,
                matches_rules=len(field_violations) == 0,
                violations=tuple(field_violations),
                confidence=field_data.get("confidence", 0.0) if field_data else 0.0,
            ))

        # Structural rules
        for srule in template.structural_rules:
            if srule.rule_type == "page_count" and request.metadata:
                page_count = request.metadata.get("page_count", 1)
                min_p = srule.parameters.get("min", 1)
                max_p = srule.parameters.get("max", 999)
                if not (min_p <= page_count <= max_p):
                    violations.append(
                        f"Page count {page_count} outside range [{min_p}, {max_p}]"
                    )

        passed = len(violations) == 0
        confidence = (
            sum(fr.confidence for fr in field_results) / len(field_results)
            if field_results else 1.0
        )

        return ValidationStep(
            step_name="structural_validation",
            passed=passed,
            details=f"{len(violations)} violations found" if violations else "All rules passed",
            field_results=tuple(field_results),
            confidence=confidence,
        )

    def _combine_results(
        self,
        request: ValidationRequest,
        template: DocumentTemplate,
        structural_result: ValidationStep,
        model_result: ValidationResult | None,
        metadata: dict[str, Any],
    ) -> ValidationResult:
        """Combine structural and model validation into a final result."""
        steps: list[ValidationStep] = [structural_result]
        notes: list[str] = []
        risk_score = 0.0

        # Structural validation contribution
        if not structural_result.passed:
            risk_score += 0.3
            notes.append(f"Structural: {len(structural_result.field_results)} field violations")

        # Model validation contribution
        is_authentic = True
        if model_result:
            steps.extend(model_result.steps)
            if model_result.risk_score > 0:
                risk_score += model_result.risk_score * 0.7
            if not model_result.is_authentic:
                is_authentic = False
                notes.append("Model detected authenticity concerns")

        # Compute overall confidence
        all_confidences = [s.confidence for s in steps if s.confidence > 0]
        overall_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        # Determine status
        risk_score = min(risk_score, 1.0)
        if risk_score >= config.RISK_INVALID_THRESHOLD:
            status = ValidationStatus.INVALID
        elif risk_score >= config.RISK_SUSPICIOUS_THRESHOLD:
            status = ValidationStatus.SUSPICIOUS
        elif not structural_result.passed:
            status = ValidationStatus.INVALID
        else:
            status = ValidationStatus.VALID

        # Generate alerts
        alerts: list[AlertPayload] = []
        if status in (ValidationStatus.INVALID, ValidationStatus.SUSPICIOUS):
            severity = AlertSeverity.CRITICAL if risk_score >= config.RISK_INVALID_THRESHOLD else AlertSeverity.WARNING
            alert_type = "fake_document" if not is_authentic else "structure_mismatch"
            alerts.append(AlertPayload(
                document_id=request.document_id,
                version=request.version,
                severity=severity,
                alert_type=alert_type,
                message=f"Document validation {status.value}: risk_score={risk_score:.2f}",
                details={
                    "matched_template": template.template_id,
                    "structural_violations": len(structural_result.field_results) - sum(
                        1 for fr in structural_result.field_results if fr.matches_rules
                    ),
                    "risk_score": risk_score,
                    "is_authentic": is_authentic,
                },
                risk_score=risk_score,
                request_id=request.request_id,
                created_at=datetime.now(timezone.utc),
            ))

        return ValidationResult(
            document_id=request.document_id,
            version=request.version,
            status=status,
            matched_template=template.template_id,
            steps=tuple(steps),
            overall_confidence=overall_confidence,
            is_authentic=is_authentic,
            risk_score=risk_score,
            alerts=tuple(alerts),
            notes=tuple(notes),
        )

    def _dispatch_alerts(
        self, result: ValidationResult, request: ValidationRequest,
    ) -> None:
        """Send alerts to the document service."""
        if not self._alert_sender:
            logger.warning("Alert sender not configured — cannot dispatch alerts")
            return

        for alert in result.alerts:
            try:
                alert_result = self._alert_sender.send_alert(alert)
                if not alert_result.sent:
                    logger.error("Failed to send alert: %s", alert_result.error)
            except Exception as exc:
                logger.exception("Error sending alert: %s", exc)

    def health(self) -> dict[str, Any]:
        """Return health status (non-blocking — no live pings)."""
        providers: dict[str, str] = {}

        # Template registry is always available (in-memory)
        providers["template_registry"] = "healthy"

        # Report adapter presence without pinging (pings are slow/blocking)
        providers["model_validator"] = "configured" if self._model_validator else "not_configured"
        providers["alert_sender"] = "configured" if self._alert_sender else "not_configured"
        providers["document_client"] = "configured" if self._document_client else "not_configured"

        # degraded if any required adapter is missing
        has_missing = (
            not self._model_validator
            or not self._alert_sender
            or not self._document_client
        )

        return {
            "status": "degraded" if has_missing else "healthy",
            "providers": providers,
            "templates_loaded": len(self._registry.list_all()),
        }
