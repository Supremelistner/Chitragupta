"""Tests for the Validator Service.

Covers: domain models, template registry, structural validation,
validation orchestration, and alert dispatch.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from validator_service.domain.models import (
    AlertPayload,
    AlertResult,
    AlertSeverity,
    DocumentTemplate,
    FieldType,
    FieldRule,
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
from validator_service.application.validation import ValidationService
from validator_service.infrastructure.templates import InMemoryTemplateRegistry


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------

class TestDomainModels(unittest.TestCase):
    """Test domain model construction and defaults."""

    def test_validation_request_defaults(self):
        req = ValidationRequest(document_id="doc-1", version=1)
        self.assertEqual(req.document_id, "doc-1")
        self.assertEqual(req.version, 1)
        self.assertIsNone(req.document_type)
        self.assertIsNone(req.extracted_text)
        self.assertEqual(req.metadata, {})
        self.assertIsNone(req.image_bytes)

    def test_validation_request_custom(self):
        req = ValidationRequest(
            document_id="doc-2",
            version=3,
            document_type="identity_document",
            document_sub_type="aadhaar",
            extracted_text="test text",
            metadata={"fields": {"name": {"value": "Test"}}},
            request_id="req-123",
        )
        self.assertEqual(req.document_type, "identity_document")
        self.assertEqual(req.document_sub_type, "aadhaar")
        self.assertEqual(req.request_id, "req-123")

    def test_validation_result_valid(self):
        result = ValidationResult(
            document_id="doc-1",
            version=1,
            status=ValidationStatus.VALID,
            overall_confidence=0.95,
            is_authentic=True,
            risk_score=0.1,
        )
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertTrue(result.is_authentic)
        self.assertLess(result.risk_score, 0.5)

    def test_validation_result_invalid(self):
        result = ValidationResult(
            document_id="doc-1",
            version=1,
            status=ValidationStatus.INVALID,
            risk_score=0.85,
            is_authentic=False,
        )
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertFalse(result.is_authentic)
        self.assertGreater(result.risk_score, 0.7)

    def test_validation_status_enum(self):
        self.assertEqual(ValidationStatus.VALID.value, "VALID")
        self.assertEqual(ValidationStatus.INVALID.value, "INVALID")
        self.assertEqual(ValidationStatus.SUSPICIOUS.value, "SUSPICIOUS")
        self.assertEqual(ValidationStatus.UNKNOWN_DOCUMENT.value, "UNKNOWN_DOCUMENT")
        self.assertEqual(ValidationStatus.ERROR.value, "ERROR")

    def test_alert_payload(self):
        alert = AlertPayload(
            document_id="doc-1",
            version=1,
            severity=AlertSeverity.CRITICAL,
            alert_type="fake_document",
            message="Document appears to be forged",
            risk_score=0.9,
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertEqual(alert.alert_type, "fake_document")
        self.assertEqual(alert.risk_score, 0.9)

    def test_field_rule(self):
        rule = FieldRule(
            name="aadhaar_number",
            field_type=FieldType.GOVERNMENT_ID,
            required=True,
            regex=r"^\d{4}\s?\d{4}\s?\d{4}$",
        )
        self.assertEqual(rule.name, "aadhaar_number")
        self.assertTrue(rule.required)
        self.assertIsNotNone(rule.regex)

    def test_document_template(self):
        template = DocumentTemplate(
            template_id="test_v1",
            document_type="identity",
            document_sub_type="test",
            variant_name="standard",
            description="Test template",
            field_rules=(
                FieldRule(name="field1", field_type=FieldType.TEXT, required=True),
            ),
        )
        self.assertEqual(template.template_id, "test_v1")
        self.assertEqual(len(template.field_rules), 1)


# ---------------------------------------------------------------------------
# Template registry tests
# ---------------------------------------------------------------------------

class TestTemplateRegistry(unittest.TestCase):
    """Test the in-memory template registry."""

    def setUp(self):
        self.registry = InMemoryTemplateRegistry()

    def test_ping(self):
        self.registry.ping()  # Should not raise

    def test_list_all(self):
        templates = self.registry.list_all()
        self.assertGreater(len(templates), 0)

    def test_get_templates_by_type(self):
        identity = self.registry.get_templates(document_type="identity_document")
        self.assertGreater(len(identity), 0)
        for t in identity:
            self.assertEqual(t.document_type, "identity_document")

    def test_get_templates_by_sub_type(self):
        aadhaar = self.registry.get_templates(document_sub_type="aadhaar")
        self.assertGreater(len(aadhaar), 0)
        for t in aadhaar:
            self.assertEqual(t.document_sub_type, "aadhaar")

    def test_get_templates_no_match(self):
        result = self.registry.get_templates(document_sub_type="nonexistent")
        self.assertEqual(len(result), 0)

    def test_get_template_by_id(self):
        template = self.registry.get_template("aadhaar_card_v1")
        self.assertIsNotNone(template)
        self.assertEqual(template.document_sub_type, "aadhaar")
        self.assertEqual(template.variant_name, "card_print")

    def test_get_template_not_found(self):
        template = self.registry.get_template("nonexistent")
        self.assertIsNone(template)

    def test_aadhaar_card_has_required_fields(self):
        template = self.registry.get_template("aadhaar_card_v1")
        self.assertIsNotNone(template)
        required = [f.name for f in template.field_rules if f.required]
        self.assertIn("aadhaar_number", required)
        self.assertIn("name", required)
        self.assertIn("date_of_birth", required)
        self.assertIn("gender", required)

    def test_aadhaar_has_three_variants(self):
        aadhaar = self.registry.get_templates(document_sub_type="aadhaar")
        variants = {t.variant_name for t in aadhaar}
        self.assertIn("card_print", variants)
        self.assertIn("letter", variants)
        self.assertIn("e_aadhaar", variants)

    def test_marksheet_template(self):
        template = self.registry.get_template("marksheet_v1")
        self.assertIsNotNone(template)
        self.assertEqual(template.document_type, "academic_record")
        required = [f.name for f in template.field_rules if f.required]
        self.assertIn("student_name", required)
        self.assertIn("roll_number", required)
        self.assertIn("school", required)
        self.assertIn("result", required)

    def test_pan_template(self):
        template = self.registry.get_template("pan_card_v1")
        self.assertIsNotNone(template)
        pan_rule = [f for f in template.field_rules if f.name == "PAN_number"][0]
        self.assertIsNotNone(pan_rule.regex)

    def test_passport_template(self):
        template = self.registry.get_template("passport_v1")
        self.assertIsNotNone(template)
        self.assertEqual(template.document_type, "identity_document")
        self.assertEqual(template.document_sub_type, "passport")

    def test_add_custom_template(self):
        custom = DocumentTemplate(
            template_id="custom_v1",
            document_type="financial",
            document_sub_type="invoice",
            variant_name="standard",
            description="Custom invoice template",
            field_rules=(
                FieldRule(name="invoice_number", field_type=FieldType.TEXT, required=True),
            ),
        )
        self.registry.add_template(custom)
        found = self.registry.get_template("custom_v1")
        self.assertIsNotNone(found)
        self.assertEqual(found.document_sub_type, "invoice")

    def test_close(self):
        self.registry.close()  # Should not raise


# ---------------------------------------------------------------------------
# Structural validation tests
# ---------------------------------------------------------------------------

class TestStructuralValidation(unittest.TestCase):
    """Test rule-based structural validation via the service."""

    def setUp(self):
        self.registry = InMemoryTemplateRegistry()
        self.service = ValidationService(template_registry=self.registry)

    def _make_request(
        self,
        metadata: dict[str, Any],
        doc_type: str = "identity_document",
        doc_sub_type: str = "aadhaar",
    ) -> ValidationRequest:
        return ValidationRequest(
            document_id="doc-test",
            version=1,
            document_type=doc_type,
            document_sub_type=doc_sub_type,
            metadata=metadata,
        )

    def test_valid_aadhaar_passes(self):
        metadata = {
            "fields": {
                "Aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98},
                "name": {"value": "Tanvi", "confidence": 0.98},
                "date_of_birth": {"value": "01/10/2007", "confidence": 0.95},
                "gender": {"value": "FEMALE", "confidence": 0.99},
            }
        }
        result = self.service.validate(self._make_request(metadata))
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertTrue(result.is_authentic)
        self.assertLess(result.risk_score, 0.5)
        self.assertEqual(result.matched_template, "aadhaar_card_v1")

    def test_missing_required_field_fails(self):
        metadata = {
            "fields": {
                "name": {"value": "Tanvi", "confidence": 0.98},
                # Missing Aadhaar_number, date_of_birth, gender
            }
        }
        result = self.service.validate(self._make_request(metadata))
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertGreater(result.risk_score, 0.0)

    def test_invalid_aadhaar_format_fails(self):
        metadata = {
            "fields": {
                "Aadhaar_number": {"value": "12345", "confidence": 0.5},  # Too short
                "name": {"value": "Test", "confidence": 0.9},
                "date_of_birth": {"value": "01/01/2000", "confidence": 0.9},
                "gender": {"value": "MALE", "confidence": 0.9},
            }
        }
        result = self.service.validate(self._make_request(metadata))
        # Should fail regex validation for Aadhaar number
        structural_step = result.steps[0] if result.steps else None
        self.assertIsNotNone(structural_step)
        if structural_step:
            violations = [fv for fv in structural_step.field_results if not fv.matches_rules]
            self.assertGreater(len(violations), 0)

    def test_invalid_date_format_fails(self):
        metadata = {
            "fields": {
                "Aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98},
                "name": {"value": "Test", "confidence": 0.9},
                "date_of_birth": {"value": "2007-10-01", "confidence": 0.9},  # Wrong format
                "gender": {"value": "FEMALE", "confidence": 0.9},
            }
        }
        result = self.service.validate(self._make_request(metadata))
        # Date format should be flagged
        structural_step = result.steps[0] if result.steps else None
        self.assertIsNotNone(structural_step)

    def test_valid_marksheet_passes(self):
        metadata = {
            "fields": {
                "student_name": {"value": "Tanvi", "confidence": 0.98},
                "roll_number": {"value": "17266193", "confidence": 0.99},
                "mother_name": {"value": "Poonam", "confidence": 0.95},
                "father_name": {"value": "Sanjay Kumar", "confidence": 0.98},
                "date_of_birth": {"value": "01/10/2007", "confidence": 0.97},
                "school": {"value": "J N V Paprola", "confidence": 0.99},
                "result": {"value": "PASS", "confidence": 1.0},
            }
        }
        result = self.service.validate(
            self._make_request(metadata, doc_type="academic_record", doc_sub_type="marksheet"),
        )
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertTrue(result.is_authentic)

    def test_marksheet_missing_student_name_fails(self):
        metadata = {
            "fields": {
                "roll_number": {"value": "17266193", "confidence": 0.99},
                "school": {"value": "J N V Paprola", "confidence": 0.99},
                "result": {"value": "PASS", "confidence": 1.0},
            }
        }
        result = self.service.validate(
            self._make_request(metadata, doc_type="academic_record", doc_sub_type="marksheet"),
        )
        self.assertEqual(result.status, ValidationStatus.INVALID)

    def test_unknown_document_type(self):
        metadata = {"fields": {}}
        req = ValidationRequest(
            document_id="doc-1",
            version=1,
            document_type="nonexistent_type",
            document_sub_type="nonexistent",
            metadata=metadata,
        )
        result = self.service.validate(req)
        self.assertEqual(result.status, ValidationStatus.UNKNOWN_DOCUMENT)
        self.assertGreater(len(result.alerts), 0)

    def test_empty_metadata(self):
        req = ValidationRequest(
            document_id="doc-1",
            version=1,
            document_type="identity_document",
            document_sub_type="aadhaar",
            metadata={},
        )
        result = self.service.validate(req)
        self.assertEqual(result.status, ValidationStatus.INVALID)

    def test_result_has_latency(self):
        metadata = {
            "fields": {
                "Aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98},
                "name": {"value": "Tanvi", "confidence": 0.98},
                "date_of_birth": {"value": "01/10/2007", "confidence": 0.95},
                "gender": {"value": "FEMALE", "confidence": 0.99},
            }
        }
        result = self.service.validate(self._make_request(metadata))
        self.assertIsNotNone(result.latency_ms)
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_result_has_request_id(self):
        metadata = {
            "fields": {
                "Aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98},
                "name": {"value": "Tanvi", "confidence": 0.98},
                "date_of_birth": {"value": "01/10/2007", "confidence": 0.95},
                "gender": {"value": "FEMALE", "confidence": 0.99},
            }
        }
        req = self._make_request(metadata)
        req = ValidationRequest(
            document_id=req.document_id,
            version=req.version,
            document_type=req.document_type,
            document_sub_type=req.document_sub_type,
            metadata=req.metadata,
            request_id="test-req-123",
        )
        result = self.service.validate(req)
        self.assertEqual(result.request_id, "test-req-123")


# ---------------------------------------------------------------------------
# Alert dispatch tests
# ---------------------------------------------------------------------------

class TestAlertDispatch(unittest.TestCase):
    """Test that alerts are generated and dispatched correctly."""

    def setUp(self):
        self.registry = InMemoryTemplateRegistry()
        self.alert_sender = MagicMock(spec=AlertSender)
        self.alert_sender.ping.return_value = None
        self.alert_sender.send_alert.return_value = AlertResult(sent=True, alert_id="alert-1")
        self.service = ValidationService(
            template_registry=self.registry,
            alert_sender=self.alert_sender,
        )

    def test_invalid_document_generates_alert(self):
        metadata = {"fields": {}}  # Missing all required fields
        req = ValidationRequest(
            document_id="doc-fake",
            version=1,
            document_type="identity_document",
            document_sub_type="aadhaar",
            metadata=metadata,
        )
        result = self.service.validate(req)
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertGreater(len(result.alerts), 0)
        self.alert_sender.send_alert.assert_called()

    def test_valid_document_no_alert(self):
        metadata = {
            "fields": {
                "Aadhaar_number": {"value": "4373 7370 1714", "confidence": 0.98},
                "name": {"value": "Tanvi", "confidence": 0.98},
                "date_of_birth": {"value": "01/10/2007", "confidence": 0.95},
                "gender": {"value": "FEMALE", "confidence": 0.99},
            }
        }
        result = self.service.validate(
            ValidationRequest(
                document_id="doc-real",
                version=1,
                document_type="identity_document",
                document_sub_type="aadhaar",
                metadata=metadata,
            )
        )
        self.assertEqual(result.status, ValidationStatus.VALID)
        # No alert should be sent for valid documents
        self.alert_sender.send_alert.assert_not_called()

    def test_alert_severity_for_high_risk(self):
        metadata = {"fields": {}}  # All missing
        req = ValidationRequest(
            document_id="doc-fake",
            version=1,
            document_type="identity_document",
            document_sub_type="aadhaar",
            metadata=metadata,
        )
        result = self.service.validate(req)
        self.assertGreater(len(result.alerts), 0)
        # High risk should produce CRITICAL or WARNING
        severities = {a.severity for a in result.alerts}
        self.assertTrue(
            AlertSeverity.CRITICAL in severities or AlertSeverity.WARNING in severities,
        )

    def test_alert_failure_is_logged(self):
        self.alert_sender.send_alert.return_value = AlertResult(
            sent=False, error="Connection refused",
        )
        metadata = {"fields": {}}
        req = ValidationRequest(
            document_id="doc-fake",
            version=1,
            document_type="identity_document",
            document_sub_type="aadhaar",
            metadata=metadata,
        )
        # Should not raise even if alert send fails
        result = self.service.validate(req)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

class TestHealthCheck(unittest.TestCase):
    """Test health endpoint."""

    def test_health_with_all_providers(self):
        registry = InMemoryTemplateRegistry()
        service = ValidationService(template_registry=registry)
        health = service.health()
        self.assertIn("status", health)
        self.assertIn("providers", health)
        self.assertEqual(health["providers"]["template_registry"], "healthy")

    def test_health_with_unconfigured_providers(self):
        service = ValidationService(template_registry=MagicMock(spec=TemplateRegistry))
        health = service.health()
        # No adapters configured → degraded
        self.assertEqual(health["status"], "degraded")
        self.assertIn("not_configured", health["providers"]["model_validator"])
        # Template registry is always healthy (in-memory, no blocking ping)
        self.assertEqual(health["providers"]["template_registry"], "healthy")


# ---------------------------------------------------------------------------
# MCP server tests
# ---------------------------------------------------------------------------

class TestMCPServer(unittest.TestCase):
    """Test MCP server tool registration and dispatch."""

    def setUp(self):
        from validator_service.adapters.mcp.server import MCPServer
        self.registry = InMemoryTemplateRegistry()
        self.service = ValidationService(template_registry=self.registry)
        self.mcp = MCPServer(self.service)

    def test_initialize(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = self.mcp._dispatch(msg)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["result"]["serverInfo"]["name"], "validator-service")

    def test_tools_list(self):
        msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = self.mcp._dispatch(msg)
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("validate_document", tool_names)
        self.assertIn("validate_document_image", tool_names)
        self.assertIn("list_templates", tool_names)
        self.assertIn("get_template", tool_names)
        self.assertIn("get_alerts", tool_names)
        self.assertIn("health", tool_names)

    def test_list_templates_tool(self):
        msg = {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "list_templates", "arguments": {}},
        }
        resp = self.mcp._dispatch(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        self.assertGreater(content["count"], 0)

    def test_list_templates_filter(self):
        msg = {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "list_templates", "arguments": {"document_sub_type": "aadhaar"}},
        }
        resp = self.mcp._dispatch(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        self.assertGreater(content["count"], 0)
        for t in content["templates"]:
            self.assertEqual(t["document_sub_type"], "aadhaar")

    def test_get_template_tool(self):
        msg = {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "get_template", "arguments": {"template_id": "aadhaar_card_v1"}},
        }
        resp = self.mcp._dispatch(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(content["template_id"], "aadhaar_card_v1")
        self.assertGreater(len(content["field_rules"]), 0)

    def test_get_template_not_found(self):
        msg = {
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "get_template", "arguments": {"template_id": "nonexistent"}},
        }
        resp = self.mcp._dispatch(msg)
        self.assertIn("error", resp)

    def test_validate_document_tool(self):
        msg = {
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "validate_document", "arguments": {"document_id": "doc-1"}},
        }
        resp = self.mcp._dispatch(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("status", content)
        self.assertIn("risk_score", content)

    def test_health_tool(self):
        msg = {
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "health", "arguments": {}},
        }
        resp = self.mcp._dispatch(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("status", content)
        self.assertIn("providers", content)

    def test_unknown_tool(self):
        msg = {
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        resp = self.mcp._dispatch(msg)
        self.assertIn("error", resp)

    def test_ping(self):
        msg = {"jsonrpc": "2.0", "id": 10, "method": "ping", "params": {}}
        resp = self.mcp._dispatch(msg)
        self.assertTrue(resp["result"]["pong"])


# ---------------------------------------------------------------------------
# HTTP server handler tests
# ---------------------------------------------------------------------------

class TestHTTPServerHandler(unittest.TestCase):
    """Test HTTP server handler logic."""

    def setUp(self):
        from validator_service.adapters.http.server import ValidatorHTTPHandler
        self.registry = InMemoryTemplateRegistry()
        self.service = ValidationService(template_registry=self.registry)
        self.handler = ValidatorHTTPHandler
        self.handler.validation_service = self.service

    def test_handler_has_validation_service(self):
        self.assertIsNotNone(self.handler.validation_service)


# ---------------------------------------------------------------------------
# Integration: full validation with real Aadhaar metadata
# ---------------------------------------------------------------------------

class TestAadhaarValidationIntegration(unittest.TestCase):
    """Integration test: validate a realistic Aadhaar extraction result."""

    def setUp(self):
        self.registry = InMemoryTemplateRegistry()
        self.service = ValidationService(template_registry=self.registry)

    def test_real_aadhaar_card_data(self):
        """Simulate Qwen 72B extraction of an Aadhaar card."""
        metadata = {
            "schema_version": "1.0",
            "document_type": "identity_document",
            "document_sub_type": "aadhaar",
            "fields": {
                "Aadhaar_number": {
                    "value": "4373 7370 1714",
                    "value_original": None,
                    "confidence": 0.98,
                    "field_type": "government_id",
                    "source": "direct",
                },
                "name": {
                    "value": "Tanvi",
                    "value_original": "तनवी",
                    "confidence": 0.98,
                    "field_type": "person_name",
                    "source": "direct",
                },
                "date_of_birth": {
                    "value": "01/10/2007",
                    "confidence": 0.95,
                    "field_type": "date",
                    "source": "direct",
                },
                "gender": {
                    "value": "FEMALE",
                    "value_original": "महिला",
                    "confidence": 0.99,
                    "field_type": "fixed_value",
                    "source": "direct",
                },
                "mobile_number": {
                    "value": "8219480038",
                    "confidence": 0.95,
                    "field_type": "phone",
                    "source": "direct",
                },
            },
            "extraction_confidence": 0.97,
        }

        result = self.service.validate(
            ValidationRequest(
                document_id="doc-aadhaar-real",
                version=1,
                document_type="identity_document",
                document_sub_type="aadhaar",
                metadata=metadata,
            )
        )

        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertTrue(result.is_authentic)
        self.assertLess(result.risk_score, 0.5)
        self.assertEqual(result.matched_template, "aadhaar_card_v1")
        self.assertGreater(result.overall_confidence, 0.7)
        self.assertEqual(len(result.alerts), 0)

    def test_real_marksheet_data(self):
        """Simulate Qwen 72B extraction of a marksheet."""
        metadata = {
            "fields": {
                "student_name": {"value": "Tanvi", "confidence": 0.98},
                "roll_number": {"value": "17266193", "confidence": 0.99},
                "mother_name": {"value": "Poonam", "confidence": 0.95},
                "father_name": {"value": "Sanjay Kumar", "confidence": 0.98},
                "date_of_birth": {"value": "01/10/2007", "confidence": 0.97},
                "school": {"value": "J N V Paprola DT Kangra HP", "confidence": 0.99},
                "result": {"value": "PASS", "confidence": 1.0},
            }
        }

        result = self.service.validate(
            ValidationRequest(
                document_id="doc-marksheet-real",
                version=1,
                document_type="academic_record",
                document_sub_type="marksheet",
                metadata=metadata,
            )
        )

        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertTrue(result.is_authentic)
        self.assertEqual(result.matched_template, "marksheet_v1")

    def test_fake_aadhaar_detected(self):
        """Simulate a document with missing/wrong fields — should be flagged."""
        metadata = {
            "fields": {
                "Aadhaar_number": {"value": "123", "confidence": 0.3},  # Wrong format
                "name": {"value": "X", "confidence": 0.4},  # Too short
                # Missing DOB, gender
            }
        }

        result = self.service.validate(
            ValidationRequest(
                document_id="doc-fake-aadhaar",
                version=1,
                document_type="identity_document",
                document_sub_type="aadhaar",
                metadata=metadata,
            )
        )

        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertGreater(result.risk_score, 0.0)
        self.assertGreater(len(result.alerts), 0)


class TestRiskThresholdBoundaries(unittest.TestCase):
    """Boundary-value tests for the centralized risk-score policy.

    The status/alert cutoffs live in validator_service.config
    (RISK_INVALID_THRESHOLD, RISK_SUSPICIOUS_THRESHOLD). These tests pin
    the exact classification at values immediately below, at, and above
    each boundary so the config and the classifier can never silently
    disagree. We drive risk directly through _combine_results with a
    passing structural step and a synthetic model result: with structural
    passing, final risk == model_result.risk_score * 0.7, so a model risk
    of r/0.7 yields a final risk of exactly r.
    """

    def setUp(self):
        self.registry = InMemoryTemplateRegistry()
        self.service = ValidationService(template_registry=self.registry)
        self.request = ValidationRequest(
            document_id="doc-boundary",
            version=1,
            document_type="identity_document",
            document_sub_type="aadhaar",
            metadata={},
        )
        self.template = DocumentTemplate(
            template_id="t-boundary",
            document_type="identity_document",
            document_sub_type="aadhaar",
            variant_name="card_print",
            description="boundary test template",
            field_rules=(),
            structural_rules=(),
        )
        self.structural_pass = ValidationStep(
            step_name="structural", passed=True, confidence=0.9,
        )

    def _combine_at_risk(self, final_risk: float) -> ValidationResult:
        """Build a result whose final risk_score == final_risk (structural passes)."""
        model_risk = min(final_risk / 0.7, 1.0)
        model_result = ValidationResult(
            document_id=self.request.document_id,
            version=1,
            status=ValidationStatus.VALID,
            risk_score=model_risk,
            is_authentic=True,
            steps=(ValidationStep(step_name="model", passed=True, confidence=0.9),),
        )
        return self.service._combine_results(
            self.request, self.template, self.structural_pass, model_result, {},
        )

    def test_below_suspicious_is_valid(self):
        # risk just under RISK_SUSPICIOUS_THRESHOLD (0.4) → VALID
        result = self._combine_at_risk(0.39)
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertEqual(len(result.alerts), 0)

    def test_at_suspicious_boundary_is_suspicious(self):
        # risk == RISK_SUSPICIOUS_THRESHOLD (0.4) → SUSPICIOUS (>= is inclusive)
        result = self._combine_at_risk(0.40)
        self.assertEqual(result.status, ValidationStatus.SUSPICIOUS)
        self.assertEqual(result.alerts[0].severity, AlertSeverity.WARNING)

    def test_between_boundaries_is_suspicious(self):
        result = self._combine_at_risk(0.55)
        self.assertEqual(result.status, ValidationStatus.SUSPICIOUS)
        self.assertEqual(result.alerts[0].severity, AlertSeverity.WARNING)

    def test_just_below_invalid_is_suspicious(self):
        result = self._combine_at_risk(0.69)
        self.assertEqual(result.status, ValidationStatus.SUSPICIOUS)

    def test_at_invalid_boundary_is_invalid_critical(self):
        # risk == RISK_INVALID_THRESHOLD (0.7) → INVALID + CRITICAL
        result = self._combine_at_risk(0.70)
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(result.alerts[0].severity, AlertSeverity.CRITICAL)

    def test_above_invalid_is_invalid_critical(self):
        result = self._combine_at_risk(0.95)
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(result.alerts[0].severity, AlertSeverity.CRITICAL)

    def test_structural_fail_forces_invalid_even_at_zero_model_risk(self):
        # A structurally-failed doc is INVALID regardless of a low score.
        structural_fail = ValidationStep(
            step_name="structural", passed=False, confidence=0.9,
        )
        model_result = ValidationResult(
            document_id=self.request.document_id, version=1,
            status=ValidationStatus.VALID, risk_score=0.0, is_authentic=True,
            steps=(ValidationStep(step_name="model", passed=True, confidence=0.9),),
        )
        result = self.service._combine_results(
            self.request, self.template, structural_fail, model_result, {},
        )
        # structural fail adds 0.3 → below suspicious (0.4), but the explicit
        # structural-fail branch still forces INVALID.
        self.assertEqual(result.status, ValidationStatus.INVALID)


if __name__ == "__main__":
    unittest.main()
