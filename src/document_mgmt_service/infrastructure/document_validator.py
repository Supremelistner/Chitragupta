"""Infrastructure — lightweight document validator for the ingestion pipeline.

Wraps the validator service's template registry, structural validation,
and temporal (date/calendar) validation logic so the document service
can validate documents during ingestion without requiring a separate
validator service process.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from validator_service.infrastructure.templates import InMemoryTemplateRegistry
from validator_service.domain.models import (
    ExpiryStatus,
    FieldType,
    TemporalRule,
    TemporalRuleType,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    ValidationStep,
    FieldValidation,
)

logger = logging.getLogger("document_mgmt_service.validator")


class IngestionValidator:
    """Validates documents during ingestion using template rules.

    This is a lightweight, in-process validator that runs structural
    validation only (no model-based visual checks).  It checks:
    - Field presence (required fields)
    - Field format (regex patterns)
    - Field constraints (length, allowed values)

    If validation fails, the document gets flagged in metadata but
    is still stored and indexed.  Alerts are logged for the document
    service to pick up.
    """

    def __init__(self) -> None:
        self._registry = InMemoryTemplateRegistry()

    def validate(
        self,
        document_id: str,
        version: int,
        metadata: dict[str, Any],
        document_type: str | None = None,
        document_sub_type: str | None = None,
    ) -> dict[str, Any]:
        """Validate a document and return validation result as a dict.

        Returns a dict with:
            - validation_status: VALID | INVALID | SUSPICIOUS | UNKNOWN_DOCUMENT
            - validation_risk_score: 0.0 - 1.0
            - validation_matched_template: template_id or None
            - validation_violations: list of violation descriptions
            - validation_alert: alert dict if validation failed, else None
        """
        extraction = metadata.get("model_extraction") or metadata
        fields = extraction.get("fields", {})
        doc_type = document_type or extraction.get("document_type")
        doc_sub_type = document_sub_type or extraction.get("document_sub_type")

        # Find matching templates
        templates = self._registry.get_templates(doc_type, doc_sub_type)
        if not templates:
            templates = self._registry.get_templates(doc_type)

        if not templates:
            return self._build_result(
                document_id, version,
                status="UNKNOWN_DOCUMENT",
                risk_score=0.0,
                matched_template=None,
                violations=[],
                alert_type="template_unknown",
                alert_message=f"No templates for {doc_type}/{doc_sub_type}",
            )

        # Pick best template
        template = self._pick_best_template(templates, fields)

        # Run structural validation
        violations: list[str] = []
        for rule in template.field_rules:
            field_data = fields.get(rule.name)
            present = field_data is not None and field_data.get("value") is not None
            value = field_data.get("value") if field_data else None

            if rule.required and not present:
                violations.append(f"Required field '{rule.name}' is missing")
                continue

            if present and value:
                if rule.regex and not re.search(rule.regex, str(value)):
                    violations.append(f"Field '{rule.name}': value '{value}' does not match pattern '{rule.regex}'")
                if rule.min_length is not None and len(str(value)) < rule.min_length:
                    violations.append(f"Field '{rule.name}': value too short ({len(str(value))} < {rule.min_length})")
                if rule.max_length is not None and len(str(value)) > rule.max_length:
                    violations.append(f"Field '{rule.name}': value too long ({len(str(value))} > {rule.max_length})")
                if rule.allowed_values and str(value) not in rule.allowed_values:
                    violations.append(f"Field '{rule.name}': value '{value}' not in allowed values")

        # Compute risk score
        total_rules = len(template.field_rules)
        required_rules = sum(1 for r in template.field_rules if r.required)
        violated = len(violations)
        risk_score = min(violated / max(required_rules, 1), 1.0)

        # ── TEMPORAL VALIDATION ──────────────────────────────────────
        temporal_status, temporal_checks, temporal_tags = self._check_temporal_validity(
            fields, template.temporal_rules,
        )

        # Temporal issues increase risk score
        if temporal_status == ExpiryStatus.EXPIRED:
            risk_score = min(risk_score + 0.3, 1.0)
            violations.append(f"Document has EXPIRED ({temporal_checks[0].get('details', '') if temporal_checks else ''})")
        elif temporal_status == ExpiryStatus.EXPIRING_SOON:
            risk_score = min(risk_score + 0.1, 1.0)
            days = temporal_checks[0].get('days_until_expiry') if temporal_checks else None
            if days is not None:
                violations.append(f"Document expires in {days} days")

        # Determine status
        if risk_score >= 0.5:
            status = "INVALID"
            alert_type = "structure_mismatch"
            severity = "CRITICAL" if risk_score >= 0.7 else "WARNING"
        elif violated > 0:
            status = "SUSPICIOUS"
            alert_type = "structure_mismatch"
            severity = "WARNING"
        else:
            status = "VALID"
            alert_type = None
            severity = None

        # Add temporal alert if expired
        if temporal_status == ExpiryStatus.EXPIRED and alert_type is None:
            alert_type = "document_expired"
            severity = "WARNING"

        return self._build_result(
            document_id, version,
            status=status,
            risk_score=risk_score,
            matched_template=template.template_id,
            violations=violations,
            alert_type=alert_type,
            alert_message=f"Validation {status}: {violated} violations" if violated else None,
            severity=severity,
            temporal_status=temporal_status.value,
            temporal_checks=temporal_checks,
            temporal_tags=temporal_tags,
        )

    def _pick_best_template(self, templates, fields):
        if len(templates) == 1:
            return templates[0]
        best = templates[0]
        best_score = -1
        extracted_count = len(fields)
        for t in templates:
            required = sum(1 for f in t.field_rules if f.required)
            score = min(required, extracted_count) / max(required, 1)
            if score > best_score:
                best_score = score
                best = t
        return best

    def _check_temporal_validity(
        self,
        fields: dict[str, Any],
        temporal_rules: tuple[TemporalRule, ...],
    ) -> tuple[ExpiryStatus, list[dict[str, Any]], list[str]]:
        """Check temporal validity of a document based on its fields and temporal rules.

        Returns:
            - overall_status: The most restrictive ExpiryStatus across all rules
            - checks: List of temporal check results
            - tags: Human-readable tags like "EXPIRED", "VALID", "NO_EXPIRY"
        """
        if not temporal_rules:
            return ExpiryStatus.UNKNOWN, [], []

        now = datetime.now(timezone.utc)
        checks: list[dict[str, Any]] = []
        tags: list[str] = []
        overall_status = ExpiryStatus.VALID  # Start optimistic

        for rule in temporal_rules:
            check = self._evaluate_temporal_rule(rule, fields, now)
            checks.append(check)
            status = check["status"]

            # Update overall status (most restrictive wins)
            if status == ExpiryStatus.EXPIRED:
                overall_status = ExpiryStatus.EXPIRED
                tags.append("EXPIRED")
            elif status == ExpiryStatus.EXPIRING_SOON and overall_status != ExpiryStatus.EXPIRED:
                overall_status = ExpiryStatus.EXPIRING_SOON
                days = check.get("days_until_expiry")
                if days is not None:
                    tags.append(f"EXPIRING_IN_{days}_DAYS")
                else:
                    tags.append("EXPIRING_SOON")
            elif status == ExpiryStatus.NO_EXPIRY:
                if overall_status == ExpiryStatus.VALID:
                    overall_status = ExpiryStatus.NO_EXPIRY
                tags.append("NO_EXPIRY")
            elif status == ExpiryStatus.UNKNOWN:
                if overall_status == ExpiryStatus.VALID:
                    overall_status = ExpiryStatus.UNKNOWN
                tags.append("TEMPORAL_UNKNOWN")

        # Deduplicate tags
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return overall_status, checks, unique_tags

    def _evaluate_temporal_rule(
        self,
        rule: TemporalRule,
        fields: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        """Evaluate a single temporal rule against document fields."""
        base = {
            "rule_type": rule.rule_type.value,
            "description": rule.description,
            "status": ExpiryStatus.UNKNOWN.value,
        }

        if rule.rule_type == TemporalRuleType.NO_EXPIRY:
            return {
                **base,
                "status": ExpiryStatus.NO_EXPIRY.value,
                "details": "Document type has no expiry",
            }

        if rule.rule_type == TemporalRuleType.FIXED_LIFETIME:
            return self._check_fixed_lifetime(rule, fields, now, base)

        if rule.rule_type == TemporalRuleType.EXPIRY_FIELD:
            return self._check_expiry_field(rule, fields, now, base)

        if rule.rule_type == TemporalRuleType.ISSUE_BASED:
            return self._check_issue_based(rule, fields, now, base)

        if rule.rule_type == TemporalRuleType.DOB_BASED:
            return self._check_dob_based(rule, fields, now, base)

        return {**base, "status": ExpiryStatus.UNKNOWN.value, "details": "Unknown rule type"}

    def _check_fixed_lifetime(self, rule, fields, now, base):
        """Check a FIXED_LIFETIME rule (e.g., passport expires 10 years after issue)."""
        # Try to find expiry date first
        expiry_field = rule.expiry_field
        if expiry_field:
            field_data = fields.get(expiry_field)
            if field_data and field_data.get("value"):
                expiry_date = self._parse_date(str(field_data["value"]))
                if expiry_date:
                    days_until = (expiry_date - now).days
                    if days_until < 0:
                        return {**base, "status": ExpiryStatus.EXPIRED.value, "days_until_expiry": days_until,
                                "expiry_date": expiry_date.isoformat(), "details": f"Expired {abs(days_until)} days ago"}
                    elif rule.warning_days and days_until <= rule.warning_days:
                        return {**base, "status": ExpiryStatus.EXPIRING_SOON.value, "days_until_expiry": days_until,
                                "expiry_date": expiry_date.isoformat(), "details": f"Expires in {days_until} days"}
                    else:
                        return {**base, "status": ExpiryStatus.VALID.value, "days_until_expiry": days_until,
                                "expiry_date": expiry_date.isoformat(), "details": f"Valid for {days_until} more days"}

        # Try to compute from issue date + lifetime
        issue_field = rule.issue_date_field
        if issue_field and rule.lifetime_days:
            field_data = fields.get(issue_field)
            if field_data and field_data.get("value"):
                issue_date = self._parse_date(str(field_data["value"]))
                if issue_date:
                    expiry_date = datetime.fromtimestamp(
                        issue_date.timestamp() + rule.lifetime_days * 86400,
                        tz=timezone.utc,
                    )
                    days_until = (expiry_date - now).days
                    if days_until < 0:
                        return {**base, "status": ExpiryStatus.EXPIRED.value, "days_until_expiry": days_until,
                                "expiry_date": expiry_date.isoformat(), "details": f"Expired {abs(days_until)} days ago"}
                    elif rule.warning_days and days_until <= rule.warning_days:
                        return {**base, "status": ExpiryStatus.EXPIRING_SOON.value, "days_until_expiry": days_until,
                                "expiry_date": expiry_date.isoformat(), "details": f"Expires in {days_until} days"}
                    else:
                        return {**base, "status": ExpiryStatus.VALID.value, "days_until_expiry": days_until,
                                "expiry_date": expiry_date.isoformat(), "details": f"Valid for {days_until} more days"}

        # Try DOB-based age check
        dob_field = rule.dob_field or "date_of_birth"
        if rule.max_age_years:
            field_data = fields.get(dob_field)
            if field_data and field_data.get("value"):
                dob = self._parse_date(str(field_data["value"]))
                if dob:
                    age_days = (now - dob).days
                    age_years = age_days / 365.25
                    if age_years > rule.max_age_years:
                        return {**base, "status": ExpiryStatus.EXPIRED.value,
                                "details": f"Age {age_years:.1f} exceeds max {rule.max_age_years} years"}
                    return {**base, "status": ExpiryStatus.VALID.value,
                            "details": f"Age {age_years:.1f} within limit"}

        return {**base, "status": ExpiryStatus.UNKNOWN.value, "details": "No expiry date or issue date found in document"}

    def _check_expiry_field(self, rule, fields, now, base):
        """Check an EXPIRY_FIELD rule — read expiry date from a specific field."""
        field_data = fields.get(rule.expiry_field)
        if not field_data or not field_data.get("value"):
            return {**base, "status": ExpiryStatus.UNKNOWN.value, "details": f"Expiry field '{rule.expiry_field}' not found"}

        field_data_value = str(field_data.get("value", ""))
        expiry_date = self._parse_date(field_data_value)
        if not expiry_date:
            return {**base, "status": ExpiryStatus.UNKNOWN.value, "details": f"Cannot parse date from '{field_data_value}'"}

        days_until = (expiry_date - now).days
        if days_until < 0:
            return {**base, "status": ExpiryStatus.EXPIRED.value, "days_until_expiry": days_until,
                    "expiry_date": expiry_date.isoformat(), "details": f"Expired {abs(days_until)} days ago"}
        elif rule.warning_days and days_until <= rule.warning_days:
            return {**base, "status": ExpiryStatus.EXPIRING_SOON.value, "days_until_expiry": days_until,
                    "expiry_date": expiry_date.isoformat(), "details": f"Expires in {days_until} days"}
        else:
            return {**base, "status": ExpiryStatus.VALID.value, "days_until_expiry": days_until,
                    "expiry_date": expiry_date.isoformat(), "details": f"Valid for {days_until} more days"}

    def _check_issue_based(self, rule, fields, now, base):
        """Check an ISSUE_BASED rule — valid from issue date, no fixed expiry."""
        issue_field = rule.issue_date_field
        if issue_field:
            field_data = fields.get(issue_field)
            if field_data and field_data.get("value"):
                issue_date = self._parse_date(str(field_data["value"]))
                if issue_date:
                    days_since = (now - issue_date).days
                    return {**base, "status": ExpiryStatus.VALID.value,
                            "issue_date": issue_date.isoformat(),
                            "details": f"Issued {days_since} days ago, no fixed expiry"}

        return {**base, "status": ExpiryStatus.UNKNOWN.value, "details": "Issue date not found in document"}

    def _check_dob_based(self, rule, fields, now, base):
        """Check a DOB_BASED rule — age-based validity."""
        dob_field = rule.dob_field or "date_of_birth"
        field_data = fields.get(dob_field)
        if not field_data or not field_data.get("value"):
            return {**base, "status": ExpiryStatus.UNKNOWN.value, "details": f"DOB field '{dob_field}' not found"}

        field_data_value = str(field_data.get("value", ""))
        dob = self._parse_date(field_data_value)
        if not dob:
            return {**base, "status": ExpiryStatus.UNKNOWN.value, "details": f"Cannot parse date '{field_data_value}'"}

        age_days = (now - dob).days
        age_years = age_days / 365.25

        if rule.max_age_years and age_years > rule.max_age_years:
            return {**base, "status": ExpiryStatus.EXPIRED.value,
                    "details": f"Age {age_years:.1f} exceeds max {rule.max_age_years} years"}

        return {**base, "status": ExpiryStatus.VALID.value,
                "details": f"Age {age_years:.1f} years"}

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse a date string in common Indian formats."""
        formats = [
            "%d/%m/%Y",    # 01/10/2007
            "%d-%m-%Y",    # 01-10-2007
            "%Y-%m-%d",    # 2007-10-01
            "%d.%m.%Y",    # 01.10.2007
            "%d %b %Y",    # 01 Oct 2007
            "%d %B %Y",    # 01 October 2007
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
        return None

    def _build_result(
        self,
        document_id, version, status, risk_score, matched_template,
        violations, alert_type=None, alert_message=None, severity=None,
        temporal_status=None, temporal_checks=None, temporal_tags=None,
    ):
        result = {
            "validation_status": status,
            "validation_risk_score": risk_score,
            "validation_matched_template": matched_template,
            "validation_violations": violations,
            "validation_temporal_status": temporal_status or "UNKNOWN",
            "validation_temporal_tags": temporal_tags or [],
            "validation_temporal_checks": temporal_checks or [],
            "validation_alert": None,
        }
        if alert_type and alert_message:
            result["validation_alert"] = {
                "document_id": document_id,
                "version": version,
                "severity": severity,
                "alert_type": alert_type,
                "message": alert_message,
                "risk_score": risk_score,
                "violations": violations,
                "temporal_status": temporal_status,
                "temporal_tags": temporal_tags,
            }
            logger.warning(
                "Validation alert: doc=%s v%s status=%s risk=%.2f violations=%d temporal=%s",
                document_id, version, status, risk_score, len(violations), temporal_status,
            )
        return result
